import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

neo4j_schema = """
Node properties:
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Genre {name: STRING}
User {name: STRING}

Relationship properties:
ACTED_IN {role: STRING}
RATED {rating: INTEGER}

The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:User)-[:RATED]->(:Movie)
(:Movie)-[:IN_GENRE]->(:Genre)
"""


# Create LLM 
t2c_llama = OllamaLLM(model_name="llama3:latest",model_params={"temperature": 0})
#t2c_llm = OpenAILLM(model_name="gpt-4.1-nano-2025-04-14", model_params={"temperature": 0})


examples = ["USER INPUT: 'Get user ratings for a movie?' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie) WHERE m.title = 'Movie Title' RETURN r.rating",
            "USER INPUT: 'Get the average user rating for a movie' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie {title: 'Movie Title'}) RETURN avg(r.rating)",
            "USER INPUT: 'Get the lowest rating for a movie' QUERY: MATCH (u:User)-[r:RATED]->(m:Movie {title: 'Movie Title'}) RETURN min(r.rating)"
            ]

# Build the retriever
retriever = Text2CypherRetriever(driver, t2c_llama, examples=examples, neo4j_schema=neo4j_schema)

llm = OpenAILLM(model_name="gpt-4o")
rag = GraphRAG(retriever=retriever, llm=llm)

query_text_1 = "What is the highest rating for Goodfellas?" # MODEL OUTPUT: The highest rating for Goodfellas is 5.0.
query_text_2 = "What is the average user rating for the movie Toy Story?" # MODEL OUTPUT: The average user rating for the movie Toy Story is 3.87
query_text_3 = "What user gives the lowest ratings?" # MODEL OUTPUT: User with userId '6' gives the lowest ratings with a rating of 0.5.

response = rag.search(
    query_text=query_text_1,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()
