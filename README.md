# Credit-Scoring-Model

This repository houses a credit risk modeling project, focusing on the development of a robust and interpretable credit scoring system.

---

## Project Structure

The project adheres to a standardized structure for clarity, maintainability, and ease of collaboration:

credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # Raw and processed data (added to .gitignore)
│   ├── raw/                   # Raw data goes here
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── init.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

---

## Credit Scoring Business Understanding

Credit scoring models are vital tools in the financial industry, enabling lenders to assess the creditworthiness of borrowers and manage credit risk effectively. This section delves into key considerations for developing such a model, particularly within a regulated environment.

### Influence of Basel II Accord on Model Interpretability and Documentation

The **Basel II Capital Accord** places significant emphasis on robust risk management and capital adequacy for financial institutions. Its core pillars (Pillar 1: Minimum Capital Requirements, Pillar 2: Supervisory Review Process, and Pillar 3: Market Discipline) directly influence the need for interpretable and well-documented credit risk models:

* **Risk Measurement Accuracy:** Basel II mandates that banks accurately measure and quantify credit risk to determine their capital requirements. This necessitates models that are not only predictive but also **transparent** in how they arrive at their predictions. An interpretable model allows financial institutions to understand the drivers of risk, validate its assumptions, and ensure its outputs align with business realities and regulatory expectations.
* **Model Validation and Auditability:** Regulatory bodies require rigorous validation of risk models. For a model to be validated effectively, its internal workings must be understandable. **Black-box models** are challenging to audit, making it difficult to justify their outputs to supervisors. Well-documented models provide a clear audit trail, explaining the data used, the methodologies applied, and the rationale behind model choices, which is crucial for regulatory compliance.
* **Pillar 2: Supervisory Review Process:** Supervisors critically assess a bank's internal capital adequacy assessment process (ICAAP). This includes evaluating the robustness of their risk models. **Interpretability** helps supervisors understand model limitations, sensitivities, and the institution's capacity to manage model risk.
* **Pillar 3: Market Discipline:** While less direct, transparent and understandable models contribute to better public disclosure of risk management practices, fostering market confidence.

In essence, Basel II transforms credit risk modeling from a purely predictive exercise into a critical regulatory and risk management function where interpretability and thorough documentation are paramount for trust, compliance, and sound decision-making.

### Necessity of Proxy Variables for Default and Associated Business Risks

In many real-world credit datasets, a direct "default" label (e.g., a formal declaration of bankruptcy or charge-off) might be scarce or unavailable, especially for newer loans or less severe delinquencies. Therefore, creating a **proxy variable** for default becomes necessary. This proxy typically involves defining default based on a combination of observable events such as:

* **Days Past Due (DPD):** A common threshold, e.g., 90 or 180 DPD.
* **Restructuring/Write-off:** Instances where the loan terms are significantly altered due to borrower distress.
* **Legal Action:** Initiation of collection procedures or legal actions.

**Why it's necessary:**

* **Scarcity of True Default Labels:** True defaults are relatively rare events, making it challenging to train robust models solely on them. Proxies expand the "default" population for model training.
* **Early Warning Signals:** A proxy can capture early signs of financial distress before a formal default occurs, allowing for proactive risk management.
* **Business Context:** The business may define "unacceptable risk" or "failure to pay" based on criteria that precede a legal default.

**Potential Business Risks of Making Predictions Based on a Proxy:**

* **Misclassification Risk:** The most significant risk is that the proxy variable may not perfectly align with the true definition of default from a business or regulatory perspective. This can lead to:
    * **False Positives (Type I Error):** Classifying a borrower as "at risk" based on the proxy when they would not have truly defaulted. This could lead to denying credit to creditworthy customers, resulting in **lost revenue** and **damaged customer relationships**.
    * **False Negatives (Type II Error):** Classifying a borrower as "low risk" when they will genuinely default. This leads to **increased loan losses** and **undermines the bank's financial stability**.
* **Model Bias:** If the proxy definition inherently biases the dataset (e.g., by excluding certain types of defaults or disproportionately capturing others), the model trained on it will inherit this bias, leading to unfair or inaccurate predictions across different customer segments.
* **Operational Misalignment:** Business processes (e.g., collections, recovery) are often tied to the formal definition of default. If the model's proxy-based predictions diverge significantly from these operational triggers, it can create inefficiencies and confusion.
* **Regulatory Scrutiny:** Regulators will scrutinize the definition and justification of any proxy variable used, and a poorly defined proxy could lead to non-compliance issues.
* **Evolving Risk Landscape:** The effectiveness of a proxy can diminish over time as economic conditions or borrower behaviors change, requiring constant re-evaluation and potential re-definition.

Therefore, careful consideration, robust validation, and continuous monitoring of the chosen proxy variable are crucial to mitigate these business risks.

### Trade-offs Between Simple Interpretable Models (Logistic Regression with WoE) and Complex High-Performance Models (Gradient Boosting)

In a regulated financial context like credit risk, choosing between model complexity and interpretability involves significant trade-offs:

| Feature/Consideration      | Simple, Interpretable Model (e.g., Logistic Regression with WoE) | Complex, High-Performance Model (e.g., Gradient Boosting) |
| :------------------------- | :--------------------------------------------------------------- | :-------------------------------------------------------- |
| **Interpretability** | **High:** Coefficients indicate the direction and magnitude of impact of each variable. WoE transformation makes relationships monotonic and more intuitive. Easy to explain to business users and regulators. | **Low:** Often considered "black-box" models. Relationships are non-linear and interactions are complex, making it difficult to understand individual variable contributions directly. |
| **Performance (Accuracy)** | **Good to Moderate:** Often performs well, especially with careful feature engineering (like WoE). May struggle with highly non-linear relationships or complex interactions. | **High:** Typically achieves state-of-the-art performance due to its ability to capture complex non-linear patterns and interactions within the data. |
| **Regulatory Compliance** | **Easier:** Meets Basel II and other regulatory requirements for transparency, validation, and explainability more readily. Less likely to face pushback during audits. | **Challenging:** Requires significant effort in explainability techniques (e.g., SHAP, LIME) to satisfy regulatory demands for understanding model decisions. Higher scrutiny from regulators. |
| **Model Validation** | **Straightforward:** Easier to validate assumptions, assess stability, and identify potential biases due to its transparent nature. | **Complex:** Validation requires advanced techniques to ensure robustness, stability, and fairness. Understanding model sensitivities can be challenging. |
| **Explainability to Stakeholders** | **Excellent:** Business users, risk managers, and even customers can generally understand why a credit decision was made. Supports clear policy formulation. | **Difficult:** Explaining individual predictions can be challenging, often requiring specialized tools and expert knowledge. Can lead to a lack of trust if decisions cannot be clearly justified. |
| **Feature Engineering** | **Crucial:** Performance heavily relies on well-engineered features (e.g., WoE, binning) to capture non-linearities and improve linearity for the model. | **Less Critical (initially):** Can discover complex interactions automatically. However, good feature engineering still improves performance and robustness. |
| **Development Time/Complexity** | Generally quicker to develop and iterate on, especially with established WoE methodologies. | Can be more computationally intensive and require more hyperparameter tuning, potentially leading to longer development cycles. |
| **Risk Management** | Easier to embed into risk management frameworks, set clear credit policies based on model outputs, and monitor performance against specific risk drivers. | More challenging to integrate directly into policy rules due to their complexity. Might require additional layers of interpretation or rule extraction. |

**Conclusion on Trade-offs:**

In a regulated financial context, the **interpretability of the model often takes precedence over marginal gains in predictive performance**. While a complex model might offer slightly higher accuracy, the inability to explain its decisions can lead to:

* **Regulatory non-compliance:** Failure to meet model validation and explainability requirements.
* **Loss of business trust:** Inability to justify credit decisions to applicants or internal stakeholders.
* **Difficulty in risk management:** Inability to understand and manage the drivers of credit risk.

Therefore, **Logistic Regression with WoE often serves as a robust baseline and a preferred choice in many production credit scoring systems** due to its inherent interpretability and ease of regulatory acceptance. Complex models like Gradient Boosting might be used for internal analytical purposes, challenger models, or in scenarios where the performance gain significantly outweighs the interpretability cost and appropriate explainability techniques can be rigorously applied and validated. The optimal choice often involves a **hybrid approach**, where interpretable models are used for core decision-making and more complex models provide additional insights or serve as benchmarks.