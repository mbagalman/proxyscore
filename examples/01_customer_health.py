"""Example: Auditing a Customer Health Score

This example demonstrates how to use proxyscore to construct and audit a simple 
customer health score based on synthetic product usage data.
"""

from proxyscore import ProxyAudit
from proxyscore.datasets import make_customer_health

def main():
    print("1. Generating synthetic customer health data...")
    # This generates a dataset of 3,000 customers over multiple months.
    df = make_customer_health(n=3000, seed=42)

    # The indicators (features) we want to use for our proxy score
    indicators = [
        "logins", 
        "feature_depth", 
        "support_tickets", 
        "nps", 
        "payment_delay_days"
    ]

    print("\n2. Running ProxyAudit...")
    # We pass the indicators, the intended downstream outcome (churn), 
    # the existing score (if any), and segments/periods for bias and drift checks.
    audit = ProxyAudit(
        indicators=df[indicators],
        score=df["health_score"],
        outcome=df["churned"],
        segments=df["segment"],
        period=df["month"],
    )
    
    report = audit.run()
    
    print("\n3. Audit Complete!")
    print(f"Overall Verdict: {report.verdict.value.upper()}")
    print("-" * 40)
    
    print("\nSummary of Checks:")
    for check_name in ["indicators", "stability", "downstream", "leakage", "segments"]:
        result = report[check_name]
        print(f"[{result.status.value.upper()}] {check_name.title()}:")
        print(f"  {result.summary}")
        
    print("\nTip: In a Jupyter Notebook, you can use `IPython.display.Markdown(report.to_markdown())` for a full rich-text breakdown.")

if __name__ == "__main__":
    main()
