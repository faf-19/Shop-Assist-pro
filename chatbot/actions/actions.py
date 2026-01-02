# shop-assist-pro/chatbot/actions/actions.py
import pickle
from pathlib import Path
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Load model from ml_model/ (go up 2 levels from chatbot/actions/)
MODEL_PATH = Path(__file__).parent.parent.parent / "ml_model" / "recommender.pkl"

# Safe loading
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    RECOMMENDER = model_data["recommender"]
    ITEM_INFO = model_data["item_info"]
    print(f"✅ [Action Server] Loaded ML model with {len(ITEM_INFO)} products")
except Exception as e:
    print(f"❌ [Action Server] Failed to load model: {e}")
    RECOMMENDER = {}
    ITEM_INFO = {}

class ActionGetProductInfo(Action):
    def name(self) -> Text:
        return "action_get_product_info"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Get the FULL user message (lowercase for matching)
        user_message = tracker.latest_message.get('text', '').lower().strip()
        
        if not user_message:
            dispatcher.utter_message(text="Could you tell me what you're looking for?")
            return []

        # Search for best-matching product using keywords
        best_match = None
        best_score = 0

        for item_id, info in ITEM_INFO.items():
            # Split keywords into individual words
            keywords = set(info.get("keywords", "").split())
            # Count how many keyword words appear in user message
            score = sum(1 for kw in keywords if kw in user_message)
            if score > best_score:
                best_score = score
                best_match = (item_id, info)

        # If no meaningful match
        if best_match is None or best_score == 0:
            dispatcher.utter_message(
                text="I couldn't find that product. Try asking about:\n"
                     "• Wireless headphones\n• Phone chargers\n• Laptop cases\n• Smartwatches"
            )
            return []

        product_id, prod = best_match
        response = f"*{prod['name']}*\n💰 ${prod['price']} • {prod['category']}"

        # Add ML-powered recommendations
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
        return []