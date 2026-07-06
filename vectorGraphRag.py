import os
import asyncio
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from hipporag import HippoRAG

from src.config import Config
from src.clients import get_clients
from src.utils import llm_model_func, embedding_func, process_pptx_file, generate_prompt
from src.rag_adapters import VectorRAGAdapter, LightRAGAdapter, HippoRAGAdapter, SpannerGraphRAGAdapter, AgenticRAGAdapter
from src.evaluation import add_eval_dataset, evaluate_retrieval

async def initialize_lightrag():
    rag = LightRAG(
        working_dir=Config.WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=768, func=embedding_func),
        graph_storage="Neo4JStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def initialize_dataframe(path: str, strategies: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for name in strategies:
        df[f"results_{name}"] = pd.Series(dtype="object")
        df[f"actual_responses_{name}"] = pd.Series(dtype="object")
    return df

async def main():
    Config.setup_directories()
    clients = get_clients()
    
    hipporag = HippoRAG(
        llm_model_name="gpt-4o-mini-2024-07-18",
        embedding_model_name="text-embedding-3-small"
        # llm_model_name=Config.LLM_MODEL,
        # llm_base_url=Config.LLM_BINDING_HOST,
        # embedding_model_name=Config.EMBEDDING_MODEL,
        # embedding_base_url=Config.EMBEDDING_BINDING_HOST,
    )
    lightrag = await initialize_lightrag()
    
    # Initialize Adapters
    adapters = {
        "vector_rag": VectorRAGAdapter(clients.qdrant_client, clients.embedding_service),
        "lightrag": LightRAGAdapter(lightrag),
        "hipporag": HippoRAGAdapter(hipporag),
        "spanner_graph": SpannerGraphRAGAdapter(
            clients.graph_store, 
            clients.embedding_service, 
            clients.llm_transformer, 
            Config.GRAPH_NAME
        ),
        "agentic_rag": AgenticRAGAdapter(clients.qdrant_client, clients.embedding_service)
        
    }


    
    df = initialize_dataframe("./Top20_BusinessCases_RAG_Benchmark.csv", list(adapters.keys()))
    contents = process_pptx_file("./202603 - NextGen Business Cases - Top 20 - IBERIA.pptx")
    
    # Generate Indexes
    indexing_intervals = {}
    for name, adapter in adapters.items():
        start_time = datetime.datetime.now()
        await adapter.index(contents)
        indexing_intervals[name] = (datetime.datetime.now() - start_time).total_seconds()
        
    index_interval_dataframe = pd.DataFrame(
        {"Indexing Interval": list(indexing_intervals.values())},
        index=list(indexing_intervals.keys())
    )
    
    # Retrieve Contexts
    retrieval_intervals = {name: [] for name in adapters.keys()}
    
    for i in range(df.shape[0]):
        query = str(df.at[i, "query"])
        await asyncio.sleep(1) # sleep for one second to avoid rate limit errors
        
        for name, adapter in adapters.items():
            start_time = datetime.datetime.now()
            context = await adapter.retrieve(query)
            retrieval_intervals[name].append((datetime.datetime.now() - start_time).total_seconds())
            
            # Using .at to safely insert lists into dataframe cells
            df.at[i, f"results_{name}"] = context
        

    retrieval_interval_dataframe_raw = pd.DataFrame(retrieval_intervals)

    retrieval_interval_dataframe = pd.DataFrame(
        {"Retrieval Interval": [sum(retrieval_intervals[name]) for name in adapters.keys()]},
        index=list(retrieval_intervals.keys())
    )
    
    interval_data = pd.concat([index_interval_dataframe, retrieval_interval_dataframe], axis=1)

    # Print out retrieved contexts
    for i in range(df.shape[0]):
        print(f"\nQuery: {df.at[i, 'query']}\n")
        for name in adapters.keys():
            print(f"results from {name}: {df.at[i, f'results_{name}']}\n\n")
        

    # Generate Responses
    async_openai_client = clients.async_openai_client
    for i in range(df.shape[0]):
        query = str(df.at[i, "query"])
        
        tasks = []
        names = []
        for name in adapters.keys():
            if name == "agentic_rag":
                df.at[i, f"actual_responses_{name}"] = adapters[name].get_response(query)
                continue
            context = df.at[i, f"results_{name}"]
            prompt = generate_prompt(query, context) # type: ignore
            names.append(name)
            tasks.append(
                async_openai_client.chat.completions.create(
                    model=Config.LLM_MODEL, # type: ignore
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
        responses = await asyncio.gather(*tasks)
        for name, response in zip(names, responses):
            df.at[i, f"actual_responses_{name}"] = response.choices[0].message.content


    # Evaluate
    strategies = list(adapters.keys())
    add_eval_dataset(df, strategies)
    await evaluate_retrieval(df, retrieval_interval_dataframe_raw, async_openai_client, strategies)

    # Plot intervals
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    plot1 = sns.barplot(x=interval_data.index, y=interval_data["Retrieval Interval"], ax=axes[0])
    axes[0].set_title("Retrieval Interval")
    plot2 = sns.barplot(x=interval_data.index, y=interval_data["Indexing Interval"], ax=axes[1])
    axes[1].set_title("Indexing Interval")
    
    plot1.set(xlabel="RAG solutions", ylabel="Seconds")
    plot2.set(xlabel="RAG solutions", ylabel="Seconds")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
