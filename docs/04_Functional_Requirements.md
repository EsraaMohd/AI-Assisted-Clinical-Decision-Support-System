# Functional Requirements

**Project:** DischargeWise

**Version:** 1.0

**Status:** Approved

---

## FR-1 Patient Risk Prediction

The system shall predict the probability of 30-day hospital readmission before patient discharge.

Priority: Must

---

## FR-2 Risk Classification

The system shall classify patients into:

- Low Risk
- Moderate Risk
- High Risk

Priority: Must

---

## FR-3 Explainable AI

The system shall display the three most influential factors contributing to each prediction.

Priority: Must

---

## FR-4 Workflow Guidance

The system shall suggest workflow-oriented clinical review actions based on the identified risk.

Priority: Must

---

## FR-5 Clinical Dashboard

The system shall present prediction results through a simple dashboard optimized for rapid clinical interpretation.

Priority: Must

---

## FR-6 Progressive Disclosure

The system shall initially display only essential information while allowing clinicians to access detailed patient information on demand.

Priority: Should

---

## FR-7 Physician Authority

The system shall clearly indicate that all clinical decisions remain the responsibility of the treating physician.

Priority: Must

---

## FR-8 Patient Search

The system shall allow clinicians to retrieve a patient assessment using a patient identifier.

Priority: Could


## Requirement Traceability

| Requirement | Source User Story |
|-------------|------------------|
| FR-1 | US-1 |
| FR-2 | US-1 |
| FR-3 | US-2 |
| FR-4 | US-3 |
| FR-5 | US-1 |
| FR-6 | US-4 |
| FR-7 | Project Scope |
| FR-8 | US-5 |