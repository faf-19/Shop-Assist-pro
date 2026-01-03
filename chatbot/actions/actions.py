# shop-assist-pro/chatbot/actions/actions.py

import pickle
from pathlib import Path
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType

# Load model from ml_model/ (go up 2 levels from chatbot/actions/)
MODEL_PATH = Path(__file__).parent.parent.parent / "ml_model" / "recommender.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    RECOMMENDER = model_data["recommender"]
    CATEGORY_KEYWORDS = model_data.get("category_keywords", {})
    ITEM_INFO = model_data["item_info"]

    print(f"✅ [Action Server] Loaded ML model with {len(ITEM_INFO)} products")
except Exception as e:
    print(f"❌ [Action Server] Failed to load model: {e}")
    RECOMMENDER = {}
    CATEGORY_KEYWORDS = {}
    ITEM_INFO = {}


class ActionGetProductInfo(Action):
    def name(self) -> Text:
        return "action_get_product_info"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:

        user_message = tracker.latest_message.get("text", "").lower().strip()

        if not user_message:
            dispatcher.utter_message(text="Could you tell me what you're looking for?")
            return []

        # -------------------------------
        # 1) Detect high-level type (phone / laptop)
        # -------------------------------
        want_phone = any(
            w in user_message
            for w in ["phone", "smartphone", "iphone", "redmi", "galaxy"]
        )
        want_laptop = any(
            w in user_message
            for w in ["laptop", "notebook"]
        )

        # -------------------------------
        # 2) Collect candidates + best match
        # -------------------------------
        best_match = None
        best_score = 0
        candidates: List[tuple] = []

        for item_id, info in ITEM_INFO.items():
            name_lower = info["name"].lower()

            # Simple type filter based on name
            if want_phone and not any(
                w in name_lower for w in ["phone", "iphone", "redmi", "galaxy"]
            ):
                continue
            if want_laptop and not any(
                w in name_lower for w in ["laptop", "notebook", "vostro", "vivobook"]
            ):
                continue

            keywords = set(info.get("keywords", "").split())
            score = sum(1 for kw in keywords if kw in user_message)

            if score > 0:
                candidates.append((item_id, info))

            if score > best_score:
                best_score = score
                best_match = (item_id, info)

        # -------------------------------
        # 3) Availability question → names only
        # -------------------------------
        is_availability_q = (
            "do you have" in user_message
            or user_message.startswith("have ")
            or user_message.endswith(" available?")
        )

        if is_availability_q and (want_phone or want_laptop):
            if not candidates:
                dispatcher.utter_message(
                    text="No, I don't have that type of product right now."
                )
                return []

            names = [info["name"] for _, info in candidates[:5]]
            names_text = "\n• " + "\n• ".join(names)

            kind = "phones" if want_phone else "laptops"
            dispatcher.utter_message(
                text=(
                    f"Yes, I have these {kind} available:"
                    f"{names_text}\n\n"
                    "Ask me about a specific one to see details."
                )
            )
            return []

        # -------------------------------
        # 4) No match at all
        # -------------------------------
        if best_match is None or best_score == 0:
            dispatcher.utter_message(
                text=(
                    "I couldn't find that product. Try asking about a specific model name "
                    "like 'iPhone 11' or 'Dell Vostro laptop'."
                )
            )
            return []

        # -------------------------------
        # 5) Specific query → full details + recommendations
        # -------------------------------
        product_id, prod = best_match
        response = f"*{prod['name']}*\n💰 ${prod['price']} • {prod.get('category', 'electronics')}"

        rec_ids = RECOMMENDER.get(product_id, [])[:2]
        if rec_ids:
            rec_list = []
            for rid in rec_ids:
                if rid in ITEM_INFO:
                    r = ITEM_INFO[rid]
                    rec_list.append(f"{r['name']} (${r['price']})")
            if rec_list:
                response += "\n\n🔍 *Frequently bought together:*\n• " + "\n• ".join(rec_list)

        dispatcher.utter_message(text=response, parse_mode="markdown")
        return [SlotSet("last_product_id", product_id)]


class ActionGetLastProductDetails(Action):
    def name(self) -> Text:
        return "action_get_last_product_details"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:

        last_product_id = tracker.get_slot("last_product_id")

        if not last_product_id or last_product_id not in ITEM_INFO:
            dispatcher.utter_message(
                text="I'm not sure which product you mean. Could you ask again?"
            )
            return []

        prod = ITEM_INFO[last_product_id]
        response = f"*{prod['name']}*\n💰 ${prod['price']} • {prod.get('category', 'electronics')}"

        rec_ids = RECOMMENDER.get(last_product_id, [])[:2]
        if rec_ids:
            rec_list = []
            for rid in rec_ids:
                if rid in ITEM_INFO:
                    r = ITEM_INFO[rid]
                    rec_list.append(f"{r['name']} (${r['price']})")
            if rec_list:
                response += "\n\n🔍 *Frequently bought together:*\n• " + "\n• ".join(rec_list)

        dispatcher.utter_message(text=response, parse_mode="markdown")
        return []
