from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import json


class GenerativeRag:
    def __init__(self, openai_api_key, engine, conversation_history,**kwargs):
        self.openai_api_key = openai_api_key
        self.engine = engine
        self.conversation_history = conversation_history
        self.chat_model = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    def generate_credit_card_recommendation(self):
        # Step 2: Prepare your credit card data
        # We will create a vectorized database of credit card details
        
        # read credit card data from a file
        # with open("credit_card_data.json", "r") as file:
        #     credit_cards = json.load(file)
        
        credit_cards = [
            {
                "name": "Chase Freedom Unlimited",
                "features": "1.5% cash back on all purchases, 5% cash back on travel purchased through Chase Ultimate Rewards, 3% cash back on dining and drugstore purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 15 months on purchases and balance transfers"
            },
            {
                "name": "Citi Double Cash Card",
                "features": "2% cash back on all purchases (1% when you buy, 1% when you pay)",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 18 months on balance transfers"
            },
            {
                "name": "Capital One Quicksilver Cash Rewards Credit Card",
                "features": "1.5% cash back on all purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 15 months"
            },
            {
                "name": "Discover it Cash Back",
                "features": "5% cash back on rotating categories, 1% cash back on all other purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 14 months on purchases and balance transfers"
            },
            {
                "name": "Bank of America Customized Cash Rewards Credit Card",
                "features": "3% cash back on a category of your choice, 2% cash back at grocery stores and wholesale clubs, 1% cash back on all other purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 15 months on purchases and balance transfers"
            }
        ]

        # Convert data to text format
        documents = [
            f"Credit Card: {card['name']}\nFeatures: {card['features']}\nEligibility: {card['eligibility']}\nBenefits: {card['benefits']}"
            for card in credit_cards
        ]

        # Step 3: Create embeddings and a vector store (e.g., FAISS)
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectorstore = FAISS.from_texts(documents, embeddings)

        # Initialize the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Set top_k to 3


        # Step 4: Set up the RAG pipeline, retrieving top 5 documents from the vector store
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            retriever=retriever,
            return_source_documents=True  # To return the credit card details used for the recommendation
            
        )


        # Step 5: Define a function to recommend credit cards based on user preferences
        def document_to_string(doc):
            return doc.page_content
        
        def recommend_credit_cards(user_query):
            result = rag_pipeline({"query": user_query})

            return {
                "recommendation": result["result"],
                "source_documents": result["source_documents"]
                #"source_documents": document_to_string(result["source_documents"][0])
            }
        
        # Example query
        # user_query = "I am looking for a credit card with low interest rates and cash back rewards."

        # transform the user query into a string
        user_query = ", ".join(f"{item['sender']}: {item['message']}" for item in self.conversation_history)
        recommendation = recommend_credit_cards(user_query)
               
        return recommendation

        # # Display the results
        # print("Recommended Credit Card(s):")
        # print(recommendation["recommendation"])

        # print("\nDetails from Source Documents:")
        # for doc in recommendation["source_documents"]:
        #     print(doc)

