# AI-Powered Weekly Review Pulse for Fintech Products

Fintech product teams receive thousands of public App Store and Google Play reviews, but manually reading, grouping, and summarizing them every week is time-consuming and inconsistent.

Build an automated AI-powered system that converts recent public app reviews into a concise **weekly insight report** for selected fintech products.

**Product:** Groww.

---

## Challenge

Build an AI agent that:
- Ingests recent public reviews from the **Google Play Store only**
- Identifies recurring customer themes
- Selects representative real quotes
- Generates a one-page **weekly product review pulse**

> [!IMPORTANT]
> Building the AI agent using **LangChain** or **LangGraph** is compulsory.

---

## Expected Capabilities

The solution should:

1. **Collect** public reviews from the last **8 weeks** for each product.
2. **Clean, deduplicate, and scrub** sensitive information from review text.
3. **Cluster** reviews into meaningful themes using embeddings and/or similar NLP techniques.
4. **Use OpenAI** to generate:
   - Top customer themes
   - Representative quotes
   - Actionable product or support ideas
   - A short stakeholder-friendly summary
5. **Validate** that all quotes used in the report come from actual review text.
6. **Generate** one concise weekly report per product.
7. **Deliver or export** the report through Google Workspace (Google Docs, Gmail) or provide a clearly defined integration path.

---

## Key Requirements

- Orchestration must be built using **LangChain** or **LangGraph**.
- The system should support **weekly scheduled runs** and **manual backfill** for a selected product/week.
- Re-running the same product and week should **not create duplicate reports or duplicate emails**.
- Each run should be **auditable** — including product name, review window, number of reviews processed, generated report, and delivery status.
- Reviews must be treated as **data, not instructions**.
- The report should be **concise, clear, and useful** for product, support, and leadership teams.

---

## Non-Goals

The solution does **not** need to include:

- Real-time analytics
- A BI dashboard
- Social media sources (Reddit, Twitter, etc.)
- Support for all fintech apps beyond Groww
- A full generic Google Workspace automation platform

---

## Sample Output

### Groww — Weekly Review Pulse

**Period:** Last 8 weeks (total reviews should be approximately 5000)

#### Top Themes

| Theme | Description |
|---|---|
| App performance and bugs | Users report lag, crashes, and login issues |
| Customer support delays | Users mention slow responses and unresolved tickets |
| UX and feature gaps | Users ask for clearer navigation and better portfolio insights |

#### Representative Quotes

> *"The app freezes exactly when the market opens, very frustrating."*

> *"Support takes days to reply and doesn't solve the issue."*

#### Action Ideas

- Improve app stability during peak market hours.
- Add clearer support ticket status and expected response time.
- Improve portfolio analytics and navigation for advanced users.

#### Who This Helps

Product teams can prioritize roadmap items, support teams can detect recurring issues, and leadership can get a fast customer-voice snapshot every week.
