import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors

BASE_DIR = "rag/documents"

# Folders matching your exact directory layout and schema requirements
DOCUMENTS_SCHEMA = {
    "Insurance policy document": [
        {
            "filename": "comprehensive_health_policy_guidelines.pdf",
            "title": "Health Insurance Policy Guidelines & Requirements",
            "sections": [
                ("a. Coverage Rules", [
                    "Inpatient hospital admissions are covered up to $100,000 per policy year.",
                    "Outpatient procedures, specialist consultations, and diagnostic testing are covered up to $10,000 per visit.",
                    "Emergency hospitalizations require formal notification within 48 hours of admission."
                ]),
                ("b. Exclusions", [
                    "Elective cosmetic surgeries, non-therapeutic treatments, and experimental medicine are strictly excluded.",
                    "Treatments performed at non-accredited facilities or unverified providers are excluded.",
                    "Claims filed more than 90 days after service date are excluded unless prior authorization was granted."
                ]),
                ("c. Claim Limits", [
                    "Standard single claim authorization cap is set at $20,000.",
                    "Maximum cumulative annual claim ceiling per policyholder is $250,000.",
                    "High-cost procedures (Cardiology, Orthopedics) exceeding $15,000 require pre-authorization."
                ]),
                ("d. Required Documentation", [
                    "All claims exceeding $5,000 require itemized hospital billing statements (CMS-1500 / UB-04 forms).",
                    "Diagnostic imaging (MRI/CT) and lab claims require attending physician clinical notes.",
                    "Prescription claims require matching pharmacy records with National Provider Identifiers (NPI)."
                ])
            ]
        }
    ],
    "Fraud investigation guidelines": [
        {
            "filename": "standard_fraud_investigation_manual.pdf",
            "title": "Fraud Risk & Investigation Guidelines",
            "sections": [
                ("a. Known Fraud Patterns", [
                    "Phantom Billing: Submitting claim statements for medical services or tests never rendered.",
                    "Upcoding: Submitting higher-severity procedure codes than the service actually rendered.",
                    "Provider Collusion: Abnormally high claim submission volumes concentrated within a single provider ID.",
                    "Rapid Claim Spikes: Multiple high-value claims submitted within 30 days of policy creation.",
                    "Unbundling: Splitting a single procedure into multiple claim components to maximize payouts."
                ]),
                ("b. Investigation Procedures", [
                    "Step 1: Extract model prediction, SHAP risk drivers, and peer group deviation ratios.",
                    "Step 2: Cross-reference provider ID against historical watch-lists and volume percentiles.",
                    "Step 3: Validate billing documentation against policy coverage limits and itemized receipts.",
                    "Step 4: Issue formal Document Request or refer case to Special Investigation Unit (SIU)."
                ]),
                ("c. Escalation Criteria", [
                    "Claims where the claim amount exceeds peer average by more than $10,000 or >200% deviation.",
                    "Policyholders with 2 or more rejected or denied claims within the preceding 12 months.",
                    "Claims associated with providers whose historical fraud rate exceeds 15%."
                ])
            ]
        }
    ],
    "Historical investigation cases": [
        {
            "filename": "historical_fraud_case_precedents.pdf",
            "title": "Historical Investigation Case Repository",
            "sections": [
                ("a. Case Description", [
                    "CASE-2024-089: High-value outpatient claim ($18,500) submitted 12 days after policy creation.",
                    "CASE-2024-112: Inpatient surgical claim exceeding peer average by $22,000 for standard appendectomy.",
                    "CASE-2025-014: Multiple outpatient claims filed across three distinct regions within 48 hours."
                ]),
                ("b. Evidence", [
                    "CASE-2024-089: Provider matched an active watch-list entity; missing itemized billing receipts.",
                    "CASE-2024-112: Hospital billing audit revealed unbundling of surgical codes and duplicate anesthesia hours.",
                    "CASE-2025-014: Patient identity theft identified; claim submission delay exceeded 60 days."
                ]),
                ("c. Investigator Decision", [
                    "CASE-2024-089: Escalated to Special Investigation Unit (SIU) for comprehensive billing audit.",
                    "CASE-2024-112: Issued partial claim adjustment and document request to hospital billing department.",
                    "CASE-2025-014: Immediate claim denial and account suspension."
                ]),
                ("d. Outcome", [
                    "CASE-2024-089: Fraud confirmed (phantom billing). Claim rejected, policy suspended, provider blacklisted.",
                    "CASE-2024-112: Operational billing error confirmed. Overbilled charges removed ($8,500 approved baseline).",
                    "CASE-2025-014: Fraud confirmed (identity theft). All claims denied and referred to legal authorities."
                ])
            ]
        }
    ],
    "Internal SOPs": [
        {
            "filename": "claims_adjudication_sops.pdf",
            "title": "Internal Standard Operating Procedures (SOP)",
            "sections": [
                ("a. When to Approve", [
                    "Model fraud probability is Normal (0) with ML risk score below 0.30.",
                    "Claim amount is within policy limits and matches peer procedure baseline averages.",
                    "All required documentation (itemized receipts, physician notes) is verified and attached."
                ]),
                ("b. When to Request Documents", [
                    "Claim risk score is elevated (0.30 to 0.65) or missing required itemized billing records.",
                    "Claim submission delay is between 30 and 90 days without emergency justification.",
                    "Claim amount exceeds $10,000 but lacks explicit physician authorization notes."
                ]),
                ("c. When to Escalate", [
                    "Model fraud probability exceeds optimal threshold (>0.65) or classifies as Suspicious/Fraud (1).",
                    "Claim exhibits known fraud patterns (upcoding, phantom billing) or high peer group deviation.",
                    "Patient profile shows 1 or more previously rejected claims in the past 12 months."
                ]),
                ("d. When to Reject", [
                    "Claim violates explicit policy exclusions (e.g., non-accredited provider, elective cosmetic care).",
                    "SIU investigation confirms fraudulent activity, altered documentation, or phantom billing.",
                    "Claim submission delay exceeds 90-day filing cutoff window without prior authorization."
                ])
            ]
        }
    ]
}

def create_styled_pdf(file_path, title, sections):
    doc = SimpleDocTemplate(file_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=15, leading=19, textColor=colors.HexColor('#1A365D'), spaceAfter=10)
    heading_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], fontSize=11, leading=14, textColor=colors.HexColor('#2B6CB0'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#2D3748'), spaceAfter=3)

    story = [
        Paragraph(title, title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10)
    ]

    for section_title, bullets in sections:
        story.append(Paragraph(section_title, heading_style))
        for bullet in bullets:
            story.append(Paragraph(f"• {bullet}", body_style))
        story.append(Spacer(1, 4))

    doc.build(story)

def main():
    print("=" * 70)
    print("GENERATING RAG DUMMY PDF DOCUMENTS")
    print("=" * 70)

    for folder_name, pdf_files in DOCUMENTS_SCHEMA.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for pdf_data in pdf_files:
            file_path = os.path.join(folder_path, pdf_data["filename"])
            create_styled_pdf(file_path, pdf_data["title"], pdf_data["sections"])
            print(f"[✓] Created PDF: {file_path}")

    print("\n[✓] All target document folders successfully populated with dummy PDFs.")

if __name__ == "__main__":
    main()