# Transformer policy: 217 observation tokens to 26 action rows.
import torch
from torch import nn
from torch.nn import functional

import actions
import features

MARKET_OP_OFFSET = len(actions.UNIT_OPS)  # Unit ops precede market ops.


class ColumnEmbedding(nn.Module):
    # Sum categorical embeddings and scaled numeric projections.

    def __init__(self, columns, scales, d_model):
        super().__init__()
        categorical = [(i, card) for i, (_, card) in enumerate(columns) if card > 0]
        numeric = [i for i, (_, card) in enumerate(columns) if card == 0]
        offsets = [sum(card for _, card in categorical[:k]) for k in range(len(categorical))]
        self.register_buffer("cat_idx", torch.tensor([i for i, _ in categorical]), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("num_idx", torch.tensor(numeric), persistent=False)
        self.register_buffer("num_scale", torch.tensor([scales.get(columns[i][0], 1.0) for i in numeric]), persistent=False)
        self.categorical = nn.Embedding(sum(card for _, card in categorical), d_model)
        self.numeric = nn.Linear(len(numeric), d_model)

    def forward(self, x):
        return self.categorical(x[..., self.cat_idx] + self.offsets).sum(-2) + self.numeric(x[..., self.num_idx].float() / self.num_scale)


class HourFormer(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8, n_dec_layers=4, ffn=2048, dropout=0.0, n_teams=0):
        super().__init__()
        self.d_model, self.n_teams = d_model, n_teams

        self.tile_embed     = ColumnEmbedding(features.TILE_COLUMNS, features.TILE_SCALE, d_model)
        self.unit_embed     = ColumnEmbedding(features.UNIT_COLUMNS, features.UNIT_SCALE, d_model)
        self.tile_pos       = nn.Embedding(features.N_TILES, d_model)
        self.board_embed    = nn.Embedding(2, d_model)
        self.unit_slot      = nn.Embedding(features.N_UNITS, d_model)
        self.global_mlp     = nn.Sequential(nn.Linear(features.GLOBAL_D, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if n_teams:
            self.team_embed = nn.Embedding(n_teams, d_model)  # Zero init, so every team starts as the pooled policy.
            nn.init.zeros_(self.team_embed.weight)

        layer               = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder        = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.enc_norm       = nn.LayerNorm(d_model)

        self.dec_pos        = nn.Embedding(actions.N_ROWS, d_model)
        self.prev_op        = nn.Embedding(1 + len(actions.UNIT_OPS) + len(actions.MARKET_OPS), d_model)
        self.prev_item      = nn.Embedding(1 + actions.N_ITEMS, d_model)
        self.prev_qty       = nn.Embedding(1 + actions.N_QTY, d_model)
        layer               = nn.TransformerDecoderLayer(d_model, n_heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder        = nn.TransformerDecoder(layer, n_dec_layers)
        self.dec_norm       = nn.LayerNorm(d_model)

        self.unit_op_head   = nn.Linear(d_model, len(actions.UNIT_OPS))
        self.market_op_head = nn.Linear(d_model, len(actions.MARKET_OPS))
        self.op_embed       = nn.Embedding(len(actions.UNIT_OPS) + len(actions.MARKET_OPS), d_model)
        self.item_head      = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, actions.N_ITEMS))
        self.item_embed     = nn.Embedding(actions.N_ITEMS, d_model)
        self.qty_head       = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, actions.N_QTY))
        self.value_head     = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

        self.register_buffer("causal_mask", torch.triu(torch.full((actions.N_ROWS, actions.N_ROWS), float("-inf")), diagonal=1), persistent=False)
        for name, table in (("unit_items", actions.UNIT_ITEMS), ("market_items", actions.MARKET_ITEMS), ("unit_needs_item", actions.UNIT_NEEDS_ITEM), ("market_needs_item", actions.MARKET_NEEDS_ITEM),
                            ("unit_needs_qty", actions.UNIT_NEEDS_QTY), ("market_needs_qty", actions.MARKET_NEEDS_QTY)):
            self.register_buffer(name, torch.tensor(table), persistent=False)

    def encode(self, tiles, units, globals_, team=None):
        # Encode observations to (B, 217, d); team -1 means unconditioned.
        batch = tiles.shape[0]
        tile_tok = (self.tile_embed(tiles.long()) + self.tile_pos.weight[None, None] + self.board_embed.weight[None, :, None]).reshape(batch, 2 * features.N_TILES, self.d_model)
        unit_tok = self.unit_embed(units.long()) + self.unit_slot.weight[None]
        global_tok = self.global_mlp(globals_.float())[:, None]
        if team is not None:
            team = team.long()
            global_tok = global_tok + (self.team_embed(team.clamp(min=0)) * (team >= 0)[:, None].to(global_tok.dtype))[:, None]
        return self.enc_norm(self.encoder(torch.cat([global_tok, tile_tok, unit_tok], 1)))

    def value(self, memory):
        return self.value_head(memory[:, 0]).squeeze(-1)

    def context(self, memory):
        # Combine observation and position embeddings for each action row.
        return torch.cat([memory[:, :1].expand(-1, actions.N_MARKET, -1), memory[:, 1 + 2 * features.N_TILES:]], 1) + self.dec_pos.weight[None]

    def previous(self, tokens):
        # Embed choices shifted right; absent fields map to index 0.
        op, item, qty = tokens.unbind(-1)
        market = torch.arange(tokens.shape[1], device=tokens.device) < actions.N_MARKET
        op = torch.where((op >= 0) & market, op + MARKET_OP_OFFSET, op)
        prev = self.prev_op((op + 1).clamp(min=0)) + self.prev_item((item + 1).clamp(min=0)) + self.prev_qty((qty + 1).clamp(min=0))
        return torch.cat([torch.zeros_like(prev[:, :1]), prev[:, :-1]], 1)

    def tables(self, op, market):
        # Return item requirements, item support, and quantity requirements.
        if market:
            return self.market_needs_item[op], self.market_items[op], self.market_needs_qty[op]
        return self.unit_needs_item[op], self.unit_items[op], self.unit_needs_qty[op]

    def log_probs(self, tiles, units, globals_, tokens, exists, temperature=1.0, team=None):
        # Return live-row log-probabilities, entropy, values, and accuracy counts.
        memory = self.encode(tiles, units, globals_, team)
        h = self.dec_norm(self.decoder(self.context(memory) + self.previous(tokens), memory, tgt_mask=self.causal_mask))

        op, item, qty = tokens.long().unbind(-1)
        m_op, u_op = op[:, :actions.N_MARKET].clamp(min=0), op[:, actions.N_MARKET:].clamp(min=0)
        live = torch.cat([op[:, :actions.N_MARKET] >= 0, (op[:, actions.N_MARKET:] >= 0) & exists.bool()], 1)
        m_logp = functional.log_softmax(self.market_op_head(h[:, :actions.N_MARKET]).float() / temperature, -1)
        u_logp = functional.log_softmax(self.unit_op_head(h[:, actions.N_MARKET:]).float() / temperature, -1)
        op_lp = torch.cat([m_logp.gather(-1, m_op[..., None]), u_logp.gather(-1, u_op[..., None])], 1).squeeze(-1) * live

        op_global = torch.cat([m_op + MARKET_OP_OFFSET, u_op], 1)
        needs_item, support, needs_qty = (torch.cat(pair, 1) for pair in zip(self.tables(m_op, True), self.tables(u_op, False)))
        needs_item, needs_qty = needs_item & live & (item >= 0), needs_qty & live & (qty >= 1)
        h_op = h + self.op_embed(op_global)
        item_logits = self.item_head(h_op).float().masked_fill(~support, float("-inf"))
        item_lp = functional.log_softmax(item_logits / temperature, -1).gather(-1, item.clamp(min=0)[..., None]).squeeze(-1)
        item_lp = torch.where(needs_item, item_lp, 0.0)  # Avoid NaNs for itemless ops.

        qty_logits = self.qty_head(h_op + self.item_embed(item.clamp(min=0))).float()
        qty_logits[..., 0] = float("-inf")  # Never emit zero quantities.
        qty_lp = torch.where(needs_qty, functional.log_softmax(qty_logits / temperature, -1).gather(-1, qty.clamp(min=0)[..., None]).squeeze(-1), 0.0)  # Missing quantities gather -inf.

        entropy = (torch.cat([-(m_logp.exp() * m_logp).sum(-1), -(u_logp.exp() * u_logp).sum(-1)], 1) * live).sum(1)
        with torch.no_grad():
            op_ok = torch.cat([m_logp.argmax(-1) == m_op, u_logp.argmax(-1) == u_op], 1) & live
            item_ok = (item_logits.argmax(-1) == item) & needs_item
            qty_ok = (qty_logits.argmax(-1) == qty) & needs_qty
            hour_ok = (op_ok | ~live).all(1) & (item_ok | ~needs_item).all(1) & (qty_ok | ~needs_qty).all(1)

        return {"log_prob": op_lp.sum(1) + item_lp.sum(1) + qty_lp.sum(1), "entropy": entropy, "value": self.value(memory),
                "market_op_correct": op_ok[:, :actions.N_MARKET].sum(), "market_op_count": live[:, :actions.N_MARKET].sum(), "unit_op_correct": op_ok[:, actions.N_MARKET:].sum(), "unit_op_count": live[:, actions.N_MARKET:].sum(),
                "item_correct": item_ok.sum(), "item_count": needs_item.sum(), "qty_correct": qty_ok.sum(), "qty_count": needs_qty.sum(), "hour_correct": hour_ok.sum()}

    forward = log_probs

    def decode_step(self, context, tokens, row, caches, prev_weights):
        # Decode one row and extend the attention cache.
        x = context[:, row:row + 1]
        if row:
            op, item, qty = tokens[:, row - 1:row].unbind(-1)
            if row - 1 < actions.N_MARKET:
                op = torch.where(op >= 0, op + MARKET_OP_OFFSET, op)
            prev = functional.embedding(torch.stack([(op + 1).clamp(min=0), (item + 1).clamp(min=0) + self.prev_op.num_embeddings, (qty + 1).clamp(min=0) + self.prev_op.num_embeddings + self.prev_item.num_embeddings], -1), prev_weights)
            x = x + (prev[..., 0, :] + prev[..., 1, :] + prev[..., 2, :])

        batch, heads = x.shape[0], self.decoder.layers[0].self_attn.num_heads
        split = lambda t: t.view(batch, 1, heads, -1).transpose(1, 2)
        for layer, (mem_k, mem_v, self_kv) in zip(self.decoder.layers, caches):
            attn = layer.self_attn
            qkv = functional.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).view(batch, 1, 3, heads, -1).permute(2, 0, 3, 1, 4)
            self_kv[:, :, :, row:row + 1].copy_(qkv[1:])
            h = functional.scaled_dot_product_attention(qkv[0], self_kv[0, :, :, :row + 1], self_kv[1, :, :, :row + 1])
            x = x + attn.out_proj(h.transpose(1, 2).reshape(batch, 1, -1))

            attn = layer.multihead_attn
            d = attn.embed_dim
            h = functional.scaled_dot_product_attention(split(functional.linear(layer.norm2(x), attn.in_proj_weight[:d], attn.in_proj_bias[:d])), mem_k, mem_v)
            x = x + attn.out_proj(h.transpose(1, 2).reshape(batch, 1, -1))

            x = x + layer._ff_block(layer.norm3(x))
        return self.dec_norm(x)[:, 0]

    @torch.no_grad()
    def act(self, tiles, units, globals_, exists, sample=False, temperature=1.0, generator=None, team=None):
        # Return tokens (B, 26, 3), log-probabilities (B,), and values (B,).
        memory = self.encode(tiles, units, globals_, team)
        batch, device = tiles.shape[0], tiles.device
        tokens = torch.full((batch, actions.N_ROWS, 3), -1, dtype=torch.long, device=device)
        log_prob = torch.zeros(batch, device=device)
        ended = torch.zeros(batch, dtype=torch.bool, device=device)
        context = self.context(memory)

        caches = []
        for layer in self.decoder.layers:
            attn = layer.multihead_attn
            d, heads = attn.embed_dim, attn.num_heads
            kv = functional.linear(memory, attn.in_proj_weight[d:], attn.in_proj_bias[d:]).view(batch, memory.shape[1], 2, heads, d // heads).permute(2, 0, 3, 1, 4)
            caches.append((kv[0], kv[1], kv.new_empty((2, batch, heads, actions.N_ROWS, d // heads))))
        prev_weights = torch.cat([self.prev_op.weight, self.prev_item.weight, self.prev_qty.weight])

        for row in range(actions.N_ROWS):
            market = row < actions.N_MARKET
            h = self.decode_step(context, tokens, row, caches, prev_weights)
            if market:
                logits, live = self.market_op_head(h).float(), ~ended
            else:
                logits, live = self.unit_op_head(h).float(), exists[:, row - actions.N_MARKET].bool()
            op, lp = choose(logits, sample, temperature, generator)
            op = torch.where(live, op, -1)
            log_prob = log_prob + lp * live
            needs_item, support, needs_qty = self.tables(op.clamp(min=0), market)
            needs_item, needs_qty = needs_item & live, needs_qty & live
            if market:
                ended = ended | (op == 0)

            h_op = h + self.op_embed(op.clamp(min=0) + (MARKET_OP_OFFSET if market else 0))
            item_logits = self.item_head(h_op).float().masked_fill(~support, float("-inf"))
            item, lp = choose(torch.where(needs_item[:, None], item_logits, 0.0), sample, temperature, generator)
            item = torch.where(needs_item, item, -1)
            log_prob = log_prob + lp * needs_item

            qty_logits = self.qty_head(h_op + self.item_embed(item.clamp(min=0))).float()
            qty_logits[..., 0] = float("-inf")
            qty, lp = choose(torch.where(needs_qty[:, None], qty_logits, 0.0), sample, temperature, generator)
            qty = torch.where(needs_qty, qty, -1)
            log_prob = log_prob + lp * needs_qty
            tokens[:, row] = torch.stack([op, item, qty], -1)

        return tokens, log_prob, self.value(memory)


def choose(logits, sample, temperature, generator):
    # Choose from logits and return the tempered log-probability.
    logp = functional.log_softmax(logits / max(temperature, 1e-6), -1)
    choice = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1) if sample else logp.argmax(-1)
    return choice, logp.gather(-1, choice[:, None]).squeeze(-1)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
