import argparse, json
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from gra.runner import gra_run_v03

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/demo")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    qa  = pipeline("text2text-generation", model=mdl, tokenizer=tok, do_sample=False)
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # transforms (symbol-preserving)
    import re
    def t_whitespace(p):  return "  " + re.sub(r"\s+"," ",p.strip()) + "  "
    def t_numbering(p):   return "1. " + p
    def t_format_q(p):    return f"Q: {p}\nA:"
    def t_parenthetical(p): return p + " (answer succinctly)."
    transforms = [t_whitespace, t_numbering, t_format_q, t_parenthetical]

    items = [
        {"id":"math_01","domain":"math","prompt":"State the Pythagorean theorem and give a numeric example."},
        {"id":"policy_01","domain":"policy","prompt":"Two benefits and two risks of LLMs in healthcare triage.",
         "restored":"Benefits:\n- faster triage [1][2]\n- 24/7 access [1]\nRisks:\n- hallucinations [2]\n- bias and fairness [3]"},
        {"id":"cnt_01","domain":"policy","prompt":"Two benefits and two risks of gauge-restored agents.",
         "restored":"Benefits:\n- invariance to rewording [1]\n- safety guardrail [1]\nRisks:\n- over-constraint [2]\n- false invariance [3]"},
        {"id":"mcq_01","domain":"mcq","prompt":"Which letter is the capital of France? A) Berlin B) Paris C) Rome D) Madrid"}
    ]

    policy_sources = {
        "policy_01":[
            "AI triage can reduce wait times and provide 24/7 access to information and basic guidance.",
            "Large language models may hallucinate clinical facts; without clinician oversight this threatens patient safety.",
            "Bias and fairness remain central risks in healthcare AI deployment, requiring monitoring and mitigation.",
            "Security vulnerabilities and data breaches are material risks for healthcare AI systems handling PHI."
        ],
        "cnt_01":[
            "Gauge-restored agents enforce invariance to rewording, providing a safety guardrail and more consistent semantics.",
            "Over-constraint may suppress recall and create false invariance that hides underlying model errors.",
            "Distribution shift and adversarial transforms can still break invariance without additional controls."
        ]
    }

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = gra_run_v03(qa, emb, transforms, items, policy_sources, str(outdir))
    print(df.to_string(index=False))
    print("\nArtifacts in", outdir.resolve())

if __name__ == "__main__":
    main()
