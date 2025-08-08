import QdrantTracker


from groupCollection import GroupCollection


def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()
    print(qdrant_tracker.all_collections())
    print(qdrant_tracker.open_collections())



if __name__ == '__main__':
    main()