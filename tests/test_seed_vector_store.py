client = get_client("./qdrant_data")

print("Total points:", client.count("papers", exact=True).count)

points, _ = client.scroll(
    collection_name="papers",
    limit=100,
    with_payload=True,
    with_vectors=False,
)

for point in points:
    print("\nID:", point.id)
    print("Paper:", point.payload["paper_id"])
    print("Chunk index:", point.payload["chunk_index"])
    print("Text:", point.payload["chunk_text"])

client.close()