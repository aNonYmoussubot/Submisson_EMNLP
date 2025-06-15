from __future__ import annotations

import os
import sys
import re
import json
import math
import dataclasses
import argparse
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_scripts.utils import eval_one 

DEEPSEEK_API_KEY = 'xxx'
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def binary_entropy(p: float, eps: float = 1e-12):
    p = max(min(p, 1 - eps), eps)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def yes_no(lp: Dict[str, float]):
    p_yes = math.exp(lp.get(" yes", float('-inf')))
    p_no  = math.exp(lp.get(" no", float('-inf')))
    z = p_yes + p_no
    return p_yes / z if z else 0.5


@dataclasses.dataclass
class Node:
    text: str
    depth: int
    p_correct: Optional[float] = None
    children: List["Node"] = dataclasses.field(default_factory=list)
    def entropy(self):
        return None if self.p_correct is None else binary_entropy(self.p_correct)
    def is_leaf(self):
        return not self.children

class Explorer:
    def __init__(self, cot_prefix: str, instruction: str, table: str, question: str,
                 temps: List[float], tau: float, max_depth: int, hes_token: str, answer: str):
        self.cot_prefix = cot_prefix
        self.instruction = instruction
        self.table = table
        self.question = question
        self.temps = temps
        self.tau = tau
        self.max_depth = max_depth
        self.hes = hes_token.lower()
        self.answer = answer

    def search(self):
        root = Node(self.cot_prefix, 0)
        self._dfs(root)
        leaves = self._leaves(root)
        if not leaves:
            return False, self.cot_prefix
        best = max(leaves, key=lambda n: n.p_correct or 0)
        solved = (best.p_correct or 0) > 0.5 or eval_one("_", best.text, self.answer)
        return solved, best.text

    def _dfs(self, n: Node):
        if n.depth >= self.max_depth:
            return
        n.p_correct = self._score(n.text)
        parent_H = n.entropy() or 1.0
        children: List[Tuple[Node, float]] = []
        for branch in ("confirm", "rethink"):
            for t in self.temps:
                cont = self._cont(n.text, branch, t, n.depth)
                if not cont:
                    continue
                child = Node(n.text + cont, n.depth + 1, p_correct=self._score(n.text + cont))
                children.append((child, 1 / (2 * len(self.temps))))
        IG = parent_H - sum(p * (c.entropy() or 0) for c, p in children)
        if IG < self.tau:
            return
        n.children = [c for c, _ in children]
        for c in n.children:
            self._dfs(c)

    def _cont(self, prefix: str, branch: str, temp: float, depth: int) -> str:
        marker = f"♦STEP{depth+1}_{branch.upper()}♦"
        directive = (
            "Please CONFIRM your current reasoning and produce final answer." if branch == "confirm"
            else "Please RETHINK thoroughly from the hesitation point and continue reasoning."
        )
        prompt = (
            f"{marker}\nCurrent reasoning so far:\n{prefix}\n\n{directive}\n"
            f"\n\"Instruction\": \"{self.instruction}\""
            f"\n\"table\": \"{self.table}\""
            f"\n\"question\": \"{self.question}\""
        )
        try:
            resp = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=4096,
            )
            reasoning = resp.choices[0].message.reasoning_content or resp.choices[0].message.content
            return f"\n{marker}\n" + reasoning
        except Exception:
            return ""
    def _score(self, chain: str) -> float:
        probe = chain + "\n\nIs the boxed answer correct? Reply Yes or No."
        try:
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": probe}],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
            )
            lp = r.choices[0].logprobs.content[0].top_logprobs 
            return yes_no(lp)
        except Exception:
            return 0.0

    def _leaves(self, n: Node):
        return [n] if n.is_leaf() else [l for c in n.children for l in self._leaves(c)]


def solve(item: Dict, temps, tau, depth, hes):
    instr, table, q, ans = item["instruction"], item["input_seg"], item["question"], item["output"]
    user_block = f"Instruction: \"{instr}\"\nTable: \"{table}\"\nQuestion: \"{q}\""
    sys_prompt = "You are an expert in table QA. Reason step by step and put final answer inside \\boxed{ }."

    for t in temps:
        try:
            comp = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_block}],
                temperature=t,
                max_tokens=4096,
            ).choices[0]
        except Exception:
            continue
        cot = comp.message.reasoning_content or ""
        if not re.search(rf".*?{hes}.*?", cot, flags=re.I | re.S):
            if eval_one("_", comp.message.content, ans):
                return True, comp.message.content
            continue
        cot_prefix = cot.split(hes, 1)[0] + hes  # include token
        explorer = Explorer(cot_prefix, instr, table, q, temps, tau, depth, hes, ans)
        ok, chain = explorer.search()
        if ok:
            return True, chain
    return False, ""

def run_dataset(args):
    src = f"train_datasets/{args.data_name}/{args.data_name}_hard_small.json"
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    solved = []
    for idx, it in tqdm(list(enumerate(data)), total=len(data)):
        ok, chain = solve(it, args.temps, args.tau, args.max_depth, args.hes_token)
        if ok:
            it["solution"] = chain
            solved.append(it)
            print(f"✓ {idx}")
        else:
            print(f"✗ {idx}")
    dst = f"train_datasets/{args.data_name}/{args.data_name}_hard_small_solved.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(solved, f, indent=2, ensure_ascii=False)
    print(f"Solved {len(solved)} / {len(data)} → {dst}")

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--data_name", default="tabfact")
    cli.add_argument("--temps", type=float, nargs="*", default=[1.0, 1.4, 1.6, 1.8])
    cli.add_argument("--tau", type=float, default=0.05)
    cli.add_argument("--max_depth", type=int, default=3)
    cli.add_argument("--hes_token", default="wait")
    cfg = cli.parse_args()
    run_dataset(cfg)
