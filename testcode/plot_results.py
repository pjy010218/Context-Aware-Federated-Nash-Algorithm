"""
plot_results.py — CAFN post-run analysis
Usage:
    python plot_results.py --result_json ./results/test_run.json
Outputs:
    - Accuracy vs α line plot
    - ΔU_i (Nash gap) bar plot
    - Pareto scatter (A vs B)
"""

import argparse, json, matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_alpha(results):
    alphas = []
    accA, accB, accC = [], [], []
    for k, v in results.items():
        if not k.startswith("alpha_"): continue
        a = float(k.split("_")[1])
        alphas.append(a)
        accA.append(v["clientA_acc"])
        accB.append(v["clientB_acc"])
        accC.append(v["clientC_acc"])
    alphas, accA, accB, accC = map(np.array, (alphas, accA, accB, accC))
    plt.figure(figsize=(7,4))
    plt.plot(alphas, accA, '-o', label='Client A')
    plt.plot(alphas, accB, '-o', label='Client B')
    plt.plot(alphas, accC, '-o', label='Client C')
    plt.xlabel("α (self-interest weight)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs α")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("accuracy_vs_alpha.png")
    print("Saved accuracy_vs_alpha.png")

def plot_nash_gap(results):
    if "nash_verification" not in results: 
        print("No NE verification info found.")
        return
    ne = results["nash_verification"]
    plt.figure(figsize=(6,4))
    plt.bar(ne.keys(), ne.values(), color=['tab:blue','tab:orange','tab:green'])
    plt.axhline(0, color='k', ls='--', lw=1)
    plt.title("ΔU_i per client (Nash gap)")
    plt.ylabel("Utility improvement if deviating")
    plt.tight_layout()
    plt.savefig("nash_gap.png")
    print("Saved nash_gap.png")

def plot_pareto(results):
    if not any(k.startswith("alpha_") for k in results): return
    vals = [(v["clientA_acc"], v["clientB_acc"]) for k,v in results.items() if k.startswith("alpha_")]
    labels = [float(k.split("_")[1]) for k in results if k.startswith("alpha_")]
    plt.figure(figsize=(6,5))
    for (a,b), lab in zip(vals, labels):
        plt.scatter(a,b, s=80, label=f"α={lab:.1f}")
        plt.text(a+0.005,b+0.005,f"{lab:.1f}",fontsize=8)
    plt.xlabel("Client A Accuracy")
    plt.ylabel("Client B Accuracy")
    plt.title("Pareto frontier: A vs B")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("pareto_AB.png")
    print("Saved pareto_AB.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_json", required=True)
    args = ap.parse_args()

    with open(args.result_json) as f:
        results = json.load(f)
    plot_accuracy_vs_alpha(results)
    plot_nash_gap(results)
    plot_pareto(results)
