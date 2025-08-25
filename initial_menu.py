import QdrantTracker


from my_collections.groupCollection import GroupCollection


def main():
    qdrant_tracker = QdrantTracker.QdrantTracker()

    initial_menu = """Select an action:
    -- "q" to quit
    -- "s" to show all collections
    -- "so" to show open collections
    -- "o" to open a collection
    -- "e" to edit a collection
    -- "c" to close a collection
    -- "d" to delete a collection
"""
    action = input(initial_menu)

    while action != "q":
        match action:
            case "s":
                print(qdrant_tracker.all_collections())
            case "so":
                print(qdrant_tracker.open_collections())
            case "o":
                collection_name = input("Name of the collection:")
                collection = qdrant_tracker.open(collection_name)
                collection.print()
            case "e":
                collection_name = input("Name of the collection:")
                collection = qdrant_tracker.get_collection(collection_name)
                collection.print()
                to_save = collection.menu()
                if to_save:
                    qdrant_tracker.save_collection(collection_name)
            case "c":
                collection_name = input("Name of the collection:")
                qdrant_tracker.disconnect(collection_name)
            case "d":
                collection_name = input("Name of the collection:")
                print("To-do")
            case _:
                print("Invalid action")
        action = input(initial_menu)



if __name__ == '__main__':
    main()