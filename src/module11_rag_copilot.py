"""Module 11: RAG-Based Inventory Copilot (Part B)."""
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from config.settings import DOCS_KB_DIR, VECTOR_STORE_PATH, OPENAI_API_KEY, OPENAI_MODEL

DEFAULT_KB_DOCS = {
    "01_reorder_policy.md": """# StockPilot AI Inventory Reorder Policy
1. Reorder Quantity Formula: Recommended Reorder Quantity = max(0, ceil((Forecast_7D_Demand * 1.2) - (Current_Inventory + Incoming_Orders))).
2. Safety Factor: A standard 1.2x (20% buffer) safety factor protects against normal daily demand volatility.
3. Inventory Coverage: Inventory Coverage (in days) equals Net Available Stock divided by Forecast Daily Demand. It measures how many days current and incoming stock will last.""",
    "02_decision_thresholds.md": """# Stockout Risk & Inventory Decision Engine Rules
- URGENT REORDER: Triggered when Stockout Probability >= 0.70 OR Inventory Coverage < 1.6 days (with positive reorder gap).
- REORDER: Triggered when Stockout Probability >= 0.50 (AT_RISK) OR Inventory Coverage < 2.6 days.
- MONITOR: Triggered when Stockout Probability is between 0.30 and 0.49 OR Inventory Coverage < 3.8 days.
- HOLD: Triggered when product is SAFE with Inventory Coverage >= 3.8 days and low stockout probability.""",
    "03_sustainability_sdg12.md": """# Sustainability Intelligence Layer & UN SDG-12 Alignment
StockPilot AI aligns with UN Sustainable Development Goal 12 (Responsible Consumption and Production) by detecting overstock (coverage > 4.2 days or Net Available > 1.15x Target Stock) and flagging potential waste risk—especially in perishable categories like Groceries. Note: Metrics are inventory-based operational indicators, not measured carbon or physical waste reductions.""",
    "04_responsible_ai_hitl.md": """# Responsible AI & Human-in-the-Loop Governance
All AI forecasts, stockout risk classifications (SAFE / AT_RISK), and reorder recommendations operate strictly as human-in-the-loop decision support. No purchase order is ever placed automatically without explicit human manager approval.""",
}


class InventoryRAGCopilot:
    """4-Stage RAG Copilot: Knowledge Base -> Chunking & Embeddings -> Vector Store -> Grounded Generation."""

    def __init__(self, kb_dir: Path = DOCS_KB_DIR, vector_path: Path = VECTOR_STORE_PATH):
        self.kb_dir = kb_dir
        self.vector_path = vector_path
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.svd = None
        self.embeddings = None

    def ensure_knowledge_base(self) -> None:
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in DEFAULT_KB_DOCS.items():
            fpath = self.kb_dir / fname
            if not fpath.exists():
                fpath.write_text(content, encoding="utf-8")

    def build_index(self) -> int:
        self.ensure_knowledge_base()
        self.chunks = []
        for fpath in sorted(self.kb_dir.glob("*.md")):
            text = fpath.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs):
                self.chunks.append({"chunk_id": f"{fpath.stem}_c{idx}", "source": fpath.name, "text": para})

        corpus = [c["text"] for c in self.chunks]
        tfidf_mat = self.vectorizer.fit_transform(corpus)
        n_comp = min(24, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1)
        if n_comp >= 2:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            dense = self.svd.fit_transform(tfidf_mat)
        else:
            dense = tfidf_mat.toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-8
        self.embeddings = dense / norms

        self.vector_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vector_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "embeddings": self.embeddings.tolist()}, f, indent=2)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            self.build_index()
        q_tfidf = self.vectorizer.transform([query])
        q_dense = self.svd.transform(q_tfidf) if self.svd is not None else q_tfidf.toarray()
        q_norm = q_dense / (np.linalg.norm(q_dense, axis=1, keepdims=True) + 1e-8)
        sims = (self.embeddings @ q_norm.T).ravel()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [{**self.chunks[i], "similarity_score": round(float(sims[i]), 4)} for i in top_idx]

    def _optional_llm_synthesis(self, query: str, context_text: str) -> str:
        if not OPENAI_API_KEY:
            return ""
        try:
            import requests
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are StockPilot AI Copilot. Answer strictly using the provided retrieved context and live inventory records."},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
                    ],
                    "temperature": 0.1,
                },
                timeout=12,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return ""

    def answer_query(self, query: str, decision_df: pd.DataFrame) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=3)
        store_match = re.search(r"\bS00[1-5]\b", query.upper())
        prod_match = re.search(r"\bP00\d{2}\b", query.upper())

        live_context_lines = []
        sub = decision_df.copy()
        if store_match:
            sub = sub[sub["Store ID"] == store_match.group(0)]
        if prod_match:
            sub = sub[sub["Product ID"] == prod_match.group(0)]

        if store_match or prod_match:
            target = sub.head(6)
            for _, r in target.iterrows():
                live_context_lines.append(
                    f"- **{r['Store ID']} / {r['Product ID']}** ({r['Category']}): Decision=`{r['Decision']}`, "
                    f"Risk=`{r['Risk_Category']}` (P={r['Stockout_Probability']:.2f}), Coverage=`{r['Inventory_Coverage_Days']:.2f}d`, "
                    f"Recommended Reorder=`{int(r['Recommended_Reorder_Qty'])}` units."
                )
        elif "at_risk" in query.lower() or "at risk" in query.lower():
            target = sub[sub["Risk_Category"] == "AT_RISK"].head(5)
            for _, r in target.iterrows():
                live_context_lines.append(
                    f"- **{r['Store ID']} / {r['Product ID']}** ({r['Category']}): Stockout Prob=`{r['Stockout_Probability']:.2f}`, "
                    f"Coverage=`{r['Inventory_Coverage_Days']:.2f}d`, Decision=`{r['Decision']}`, Reorder=`{int(r['Recommended_Reorder_Qty'])}` units."
                )

        doc_summary = "\n".join([f"- **[{c['source']}]** {c['text']}" for c in retrieved])
        combined_context = doc_summary + ("\n" + "\n".join(live_context_lines) if live_context_lines else "")

        llm_reply = self._optional_llm_synthesis(query, combined_context)
        if llm_reply:
            final_answer = llm_reply
        else:
            parts = [f"### Grounded Policy & System Documentation\n{doc_summary}"]
            if live_context_lines:
                parts.append("### Matching Live Inventory Records\n" + "\n".join(live_context_lines))
            parts.append("*Human-in-the-loop governance: All recommendations require manager approval prior to purchase order execution.*")
            final_answer = "\n\n".join(parts)

        return {
            "query": query,
            "answer": final_answer,
            "retrieved_chunks": retrieved,
            "live_records_matched": len(live_context_lines),
        }
