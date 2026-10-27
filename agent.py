# Local inference with D1+A1+W1.
import argparse
import json
import sys

from rules import Rules, parse


class Agent:
    def __init__(self, policy, rules="D1+A1+W1"):
        self.policy = policy
        self.rule_set = Rules(parse(rules))

    def __call__(self, obs, configuration=None):
        view = self.rule_set.observe(obs)
        action = self.policy.act([view], [int(obs["player"])])[0]
        return self.rule_set(obs, action)


def load_agent(checkpoint, device="cuda", rules="D1+A1+W1"):
    from workers import Policy, load_model

    model, _ = load_model(checkpoint, device)
    return Agent(Policy(model, device), rules)


def main():
    parser = argparse.ArgumentParser(description="Read JSON observations from stdin and write JSON actions.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rules", default="D1+A1+W1")
    args = parser.parse_args()
    agent = None
    for line in sys.stdin:
        request = json.loads(line)
        command = request.get("cmd")
        if command == "exit":
            break
        if command == "ping":
            reply = {"pong": True}
        else:
            if agent is None:
                agent = load_agent(args.checkpoint, args.device, args.rules)
            reply = {"action": agent(request.get("obs", request), request.get("config"))}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
