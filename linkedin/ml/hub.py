# linkedin/ml/hub.py
"""Campaign kit: download from HuggingFace, lazy-load, freemium campaign import."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_cached_kit: Optional[dict] = None
_cache_attempted = False


# ------------------------------------------------------------------
# Kit download & loading
# ------------------------------------------------------------------

_DEFAULT_REPO_ID = "eracle/campaign-kit"


def download_kit(revision: str = "v1") -> Optional[Path]:
    """Download campaign kit from HuggingFace Hub to a temp directory. Returns path or None."""
    try:
        import huggingface_hub
        from huggingface_hub import snapshot_download

        huggingface_hub.utils.disable_progress_bars()
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)

        kit_dir = Path(tempfile.mkdtemp(prefix="openoutreach-kit-"))
        path = snapshot_download(
            repo_id=_DEFAULT_REPO_ID,
            revision=revision,
            local_dir=str(kit_dir),
        )
        # Remove HF download metadata cache — not needed after download
        shutil.rmtree(kit_dir / ".cache", ignore_errors=True)
        logger.info("[Freemium] Kit downloaded to %s", path)
        return Path(path)
    except Exception:
        logger.info("[Freemium] Kit download failed", exc_info=True)
        return None


def load_kit_config(kit_dir: Path) -> Optional[dict]:
    """Parse config.json from kit directory. Returns dict or None."""
    _OBJECTIVE = """
# LinkedIn Leads – Plabs AI Services
This document outlines the target customer and keyword strategy for acquiring new clients for Plabs, a provider of scalable, production-ready AI solutions designed to solve real business challenges.
## Ideal Customer Profile: The AI-Driven Transformer
Our ideal customer is the "AI-Driven Transformer"—a forward-thinking decision-maker in mid-sized to large organizations who is actively exploring how Artificial Intelligence can create measurable business impact. They are typically CTOs, CIOs/DSIs, Heads of Data/AI, CMOs, or Innovation Leaders responsible for driving digital transformation and operational efficiency.
## Key Characteristics:
- Business-First, Not Tech-First: They are not interested in AI for hype—they seek concrete use cases that reduce costs, increase revenue, or improve efficiency. Every initiative must tie directly to business outcomes
- Innovation-Oriented but Pragmatic: They want to leverage cutting-edge technologies like Generative AI and machine learning but require structured, reliable partners who can bring ideas into production—not just prototypes
- Data-Aware but Execution-Constrained: They understand the value of their data but often lack the internal resources, expertise, or infrastructure to fully operationalize AI at scale
- Efficiency & Automation Focused: They are under pressure to optimize operations, reduce manual processes, and improve decision-making through intelligent automation
- Scalability & Integration Mindset: They need solutions that integrate seamlessly into existing systems and can scale across teams, departments, or geographies
- Long-Term Value Seekers: They prioritize partners who can support them beyond initial delivery, ensuring continuous improvement, monitoring, and evolution of AI systems.
- They are transforming their organizations into data-driven, intelligent businesses and need a partner who can deliver robust, scalable AI solutions from strategy to production.
## LinkedIn Search Keywords
To identify and connect with these profiles, we target a mix of roles, technological focus areas, and expressed business challenges.
1. Job Titles & Roles
- Chief Technology Officer (CTO)
- Chief Information Officer (CIO) / DSI
- Chief Data Officer (CDO)
- Head of Data
- Head of AI / AI Lead
- Machine Learning Engineer (Lead/Senior)
- Head of Innovation
- Chief Marketing Officer (CMO)
- Digital Transformation Director
- VP Engineering
- Product Director / Head of Product
- Directeur technique (CTO)
- Directeur des systèmes d'information (DSI)
- Directeur des données (CDO)
- Responsable Data
- Responsable IA
- Ingénieur Machine Learning senior
- Directeur de l’innovation
- Directeur de la transformation digitale
- Directeur marketing (CMO)
- Directeur de l’ingénierie
- Directeur produit
2. Interests & Technologies
- Artificial Intelligence (AI)
- Machine Learning
- Generative AI / LLMs
- Natural Language Processing (NLP)
- Computer Vision
- Data Engineering
- MLOps
- Cloud Computing (AWS, GCP, Azure)
- Data Platforms / Data Lakes
- Automation / RPA
- SaaS / Digital Products
- Intelligence artificielle (IA)
- Apprentissage automatique (Machine Learning)
- IA générative / LLMs
- Traitement du langage naturel (NLP)
- Vision par ordinateur
- Ingénierie des données
- MLOps
- Cloud computing (AWS, GCP, Azure)
- Plateformes de données / Data Lakes
- Automatisation / RPA
- SaaS / Produits digitaux
3. Pain Points & Goals (keywords in their posts or summaries)
- "AI use cases"
- "Generative AI applications"
- "automation des processus" / "process automation"
- "data-driven decision making"
- "scaling AI in production"
- "MLOps"
- "document automation"
- "customer experience personalization"
- "cost reduction through AI"
- "time-to-market acceleration"
- "legacy system integration"
- "data pipeline challenges"
- Cas d’usage de l’IA
- Applications d’IA générative
- Automatisation des processus
- Prise de décision basée sur les données
- Passage à l’échelle de l’IA en production
- MLOps
- Automatisation documentaire
- Personnalisation de l’expérience client
- Réduction des coûts grâce à l’IA
- Accélération du time-to-market
- Intégration des systèmes legacy
- Défis des pipelines de données
    """
    
    _PRODUCT_DOCS = """# AI Services by Plabs
## Overview
At Plabs, we help organizations unlock the full potential of Artificial Intelligence by delivering scalable, production-ready solutions tailored to real business challenges.
Our AI services combine deep technical expertise with business-first thinking, enabling companies to innovate faster, reduce operational costs, and create intelligent digital experiences.
## Our AI Capabilities
### 1. Machine Learning Solutions
We design, build, and deploy custom machine learning models that transform data into actionable insights.
**What we offer:**
* Predictive analytics and forecasting
* Recommendation systems
* Classification and clustering models
* Fraud detection and anomaly detection
### 2. Generative AI & LLM Applications
We develop cutting-edge applications powered by large language models and generative AI.
**Use cases:**
* Chatbots and virtual assistants
* Content generation (text, code, summaries)
* Knowledge assistants and copilots
* Document processing and automation
### 3. Natural Language Processing (NLP)
Turn unstructured text into structured value with advanced NLP solutions.
**Capabilities:**
* Sentiment analysis
* Entity recognition
* Text summarization
* Semantic search
### 4. Computer Vision
We enable machines to interpret and act on visual data.
**Applications:**
* Image and video recognition
* Object detection
* Facial recognition
* Quality inspection systems
### 5. AI-Powered Automation
Streamline workflows and reduce manual effort with intelligent automation.
**Solutions include:**
* Process automation (RPA + AI)
* Intelligent document processing
* Workflow optimization
### 6. Data Engineering & AI Infrastructure
Robust AI systems start with strong data foundations.
**We provide:**
* Data pipelines and ETL processes
* Data lakes and warehouses
* Model deployment (MLOps)
* Cloud-based AI infrastructure
## Our Approach
1. **Discovery & Strategy**
   We identify high-impact AI opportunities aligned with your business goals.
2. **Design & Prototyping**
   Rapid experimentation to validate ideas and reduce risk.
3. **Development & Integration**
   End-to-end implementation with seamless system integration.
4. **Deployment & Scaling**
   Production-ready solutions with performance monitoring.
5. **Continuous Improvement**
   Ongoing optimization using real-world feedback and data.
## 🏭 Industries We Serve
* Fintech
* Healthcare
* Retail & eCommerce
* Manufacturing
* Logistics & Supply Chain
* Media & Entertainment
## Why Plabs?
* Top 1% engineering talent
* Proven track record with global clients
* Scalable, nearshore delivery model
* Flexible engagement models
* Strong expertise across AI, cloud, and data
## Business Outcomes
Our AI solutions help clients:
* Reduce operational costs
* Improve decision-making
* Increase revenue through personalization
* Enhance customer experience
* Accelerate time-to-market"""

    _FOLLOW_UP_TEMPLATE = """You are a professional networker reaching out on LinkedIn.
Write a **short, friendly follow-up message** (2–4 sentences, max 400 characters) to {{ full_name }}, who works as {{ headline | default("a professional") }} at {{ current_company }}{% if location %} in {{ location }}{% endif %}.
{% if product_description %}
About the product/service:
{{ product_description }}
{% endif %}
Reference:
- Their recent role or company
- A shared connection (we have {{ shared_connections }} in common) if > 0
- Naturally mention how our product/service could be relevant to them, without being salesy
- Keep it natural, warm, and professional
- End with a soft call-to-action (e.g., coffee chat, quick call, or "happy to connect further")
- Avoid keep variables or placeholder like [Shared Connection\'s Name] or [Your Name]. Craft a message that the recipient can directly read without any additional processing.
Do **NOT** use placeholders in the output.
Do **NOT** use placeholders in the output like the following:
[Your Name]
[Your Contact Information]
[Your Company]
Do **NOT** sign the message. Do NOT add any name, signature, closing name, or sign-off name at the end.
The message must end with the call-to-action sentence itself — never with a name like "Best, John" or "— Sarah" or similar.
"""
    config = {
        "action_fraction": 0.2,
        "product_docs": _PRODUCT_DOCS,
        "campaign_objective": _OBJECTIVE,
        "booking_link": None,
        "followup_template": _FOLLOW_UP_TEMPLATE,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "seed_profiles": [],
    }
    return config
    # try:
    #     config_path = kit_dir / "config.json"
    #     data = json.loads(config_path.read_text())

    #     required = ("action_fraction", "product_docs", "campaign_objective",
    #                  "booking_link")
    #     for key in required:
    #         if key not in data:
    #             logger.info("[Freemium] Kit config missing key: %s", key)
    #             return None

    #     logger.info("[Freemium] Kit config loaded (action_fraction=%.2f)", data["action_fraction"])
    #     return data
    # except Exception:
    #     logger.info("[Freemium] Kit config load failed", exc_info=True)
    #     return None


def load_kit_model(kit_dir: Path):
    """Load pre-trained model from kit. Returns any sklearn-compatible estimator or None.

    The loaded object just needs a ``predict(X)`` method — it can be a
    Pipeline, a bare estimator, or any future model architecture.
    """
    try:
        import joblib

        model = joblib.load(kit_dir / "model.joblib")

        if not hasattr(model, "predict"):
            logger.info("[Freemium] Kit model has no predict() method")
            return None

        logger.info("[Freemium] Kit model loaded (%s)", type(model).__name__)
        return model
    except Exception:
        logger.info("[Freemium] Kit model load failed", exc_info=True)
        return None


def fetch_kit() -> Optional[dict]:
    """Lazy-load and cache the kit. Returns {"config": ..., "model": ...} or None."""
    global _cached_kit, _cache_attempted

    if _cache_attempted:
        return _cached_kit

    _cache_attempted = True

    kit_dir = download_kit()
    if kit_dir is None:
        return None

    config = load_kit_config(kit_dir)
    if config is None:
        return None

    model = load_kit_model(kit_dir)
    if model is None:
        return None

    _cached_kit = {"config": config, "model": model}
    return _cached_kit
