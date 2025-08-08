import QdrantTracker


from my_collections.groupCollection import GroupCollection


def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()
    collection = GroupCollection.download_qdrant_collection("testing_w_groups", qdrant_tracker)

    print(qdrant_tracker.all_collections())
    print(qdrant_tracker.open_collections())



if __name__ == '__main__':
    main()