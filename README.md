# OfferFlow™ – Intelligent Loyalty Offer Recommendation Engine

OfferFlow™ is a data-driven loyalty offer recommendation app designed for retail businesses with customer lifecycle segmentation (e.g., Early Life, In-Life, Lapsed). It provides smart, personalized campaign recommendations, monitors campaign effectiveness, and offers diagnostic insights into offer strategy.

---

## 🚀 Key Features

- **Campaign Effectiveness Dashboard**: Track achievement rate, ROI, redemption, and visit uplift by subcategory, segment, region, and offer type.
- **Offer Simulator**: Simulate scenarios with generosity %, duration, and segment targeting to forecast sales, profit uplift, and cost.
- **Quality Control Panel**: Automatically flags offer timing, eligibility, collision, and generosity violations across customer journeys.
- **Customer Journey Migration**: Visualize new acquisition, reactivation, and lapses over time.
- **AI Insights (OpenAI/Gemini)**: Natural language interface to ask questions about offer performance and receive instant visual + textual analysis.

---

## 📂 Project Structure

```
├── app.py                        # Streamlit app entry point
├── requirements.txt             # All required dependencies
├── Corrected_Offer_Data.csv     # Sample cleaned offer dataset
├── Simulation_Scenarios.csv     # Simulated scenario KPIs
├── .streamlit/secrets.toml      # API Keys (handled in deployment settings)
└── README.md                    # This file
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/offerflow-loyalty-app.git
cd offerflow-loyalty-app
```

### 2. Install Requirements

Make sure you have Python 3.8+ installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run app.py
```

---

## 🔐 Configuring Secrets

Add your Gemini API key inside Streamlit secrets (when deploying via Streamlit Cloud):

```toml
[generativeai]
GOOGLE_API_KEY = "your-api-key-here"
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push code and data files to your GitHub repo.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → New App → Connect your repo.
3. Set `app.py` as the entry point.
4. Configure your secrets (Gemini API key) under **Settings → Secrets**.
5. Deploy and share the link.

---

## 🧠 Tech Stack

- **Streamlit** – UI framework
- **Pandas / Numpy** – Data manipulation
- **Plotly** – Interactive charting
- **OpenAI / Gemini** – AI-powered insights

---

## 💼 Example Use Cases

- Retail chains managing loyalty across thousands of customers
- Fuel + Convenience stores optimizing footfall through offers
- FMCG brand campaign planning and post-evaluation
- Offer compliance and diagnostics across multiple customer journeys

---

## 📧 Feedback or Suggestions?

Feel free to raise an issue or suggest enhancements through GitHub issues.

---

© 2025 OfferFlow™ – Built for modern retail intelligence.