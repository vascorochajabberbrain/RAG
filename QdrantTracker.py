from qdrant_client import QdrantClient, models
import os

from my_collections.SCS_Collection import SCS_Collection
from my_collections.groupCollection import GroupCollection
from my_collections.groupCollection_sameSource import groupCollection_sameSource

COLLECTIONS_TYPES_MAP = {
    "group": GroupCollection,
    "group_sameSource": groupCollection_sameSource,
    "scs": SCS_Collection
}


class QdrantTracker:
    """
    A class to track Qdrant connection and open collections.
    """

    def __init__(self):
        print("QdrantTracker: Initializing QdrantTracker...")
        try:
            self._connection = QdrantClient(
                url = os.getenv("QDRANT_URL"),
                api_key = os.getenv("QDRANT_API_KEY"),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Qdrant: {e}. "
                "Please check your QDRANT_URL and QDRANT_API_KEY environment variables."
            )
        self._open_collections = set()
        print("QdrantTracker: QdrantTracker initialized.")

    def open(self, collection_name):
        """
        Make it sure that exists a 1:1 relation between Qdrant and local collection.
        Will certify if the user wants to use an existing collection or create a new one.
        In case of an existing collection it can return the points qDrant points from it.
        It is of each collection to know how to handle the points.
        If the collection does not exist, it will create a new one.
        Returns the collection name and a list of points if the collection exists.
        """
        if collection_name is None:
            collection_name = input("Insert the name of the collection:")

        points = []
        # Loop until we have a valid collection name
        while True:

            # we assume that if the collection does not exist, the user wants to create a new collection
            if not self._existing_collection_name(collection_name):
                print(f"QdrantTracker: Collection {collection_name} does not exist. Going to execute the new method...")
                collection =self.new(collection_name)  # Create a new collection
                return collection

            # if the collection exists, we make sure the user wants to overwrite it
            else:

                #loop until the user gives a valid answer
                while True:
                    using = input(f"The collection {collection_name} already exists. Do you want to use the existing points? (y/n): ")
                    if using.lower() in ['y', 'n']:
                        break
                    else:
                        print("Please enter 'y' for yes or 'n' for no.")

                if using.lower() == 'n':
                    print(f"Qdrant: Deleting collection {collection_name}...")
                    self._delete_collection(collection_name)
                    print(f"QdrantTracker: Collection {collection_name} does not exist. Going to execute the new method...")
                    collection =self.new(collection_name)  # Create a new collection
                    return collection

                elif using.lower() == 'y':
                    print(f"Qdrant: Getting points from collection {collection_name}...")
                    points = self._get_all_points_payload(collection_name, points)
                    
                    break

        try:
            collection_type = self.get_collection_type(points[0]) #any new collection will have collection information on any point
        except:#scenario when it is an old collection without collection information on any point
            collection_type = input(f"""From the options: {', '.join(COLLECTIONS_TYPES_MAP.keys())}\nEnter the type of the collection you want: """)
            while collection_type not in COLLECTIONS_TYPES_MAP:
                print(f"QdrantTracker: Invalid collection type.")
                collection_type = input(f"""From the options: {', '.join(COLLECTIONS_TYPES_MAP.keys())}\nEnter the type of the collection you want: """)

        collection = COLLECTIONS_TYPES_MAP[collection_type].init_from_qdrant(collection_name, points)
        self._open_collections.add(collection)
        print(f"QdrantTracker: Collection: {collection_name} is open.")
        return collection
    
    def new(self, collection_name=None, collection_type=None):
        """
        Create a new collection with the given name.
        If collection_type is None, prompts user for type (interactive).
        For workflow/API use, pass collection_type to avoid prompts (e.g. 'scs' or 'group').
        """
        if collection_name is None:
            collection_name = input("Insert the name of the collection:")

        if self._existing_collection_name(collection_name):
            print(f"QdrantTracker: Collection {collection_name} already exists. Going to execute the open method...")
            return self.open(collection_name)

        if collection_type is None:
            while True:
                collection_type = input(f"""From the options: {', '.join(COLLECTIONS_TYPES_MAP.keys())}\nEnter the type of the collection you want: """)
                if collection_type not in COLLECTIONS_TYPES_MAP:
                    print("QdrantTracker: Invalid collection type.")
                    continue
                break
        elif collection_type not in COLLECTIONS_TYPES_MAP:
            raise ValueError(f"Invalid collection type: {collection_type}. Must be one of {list(COLLECTIONS_TYPES_MAP.keys())}")

        print(f"QdrantTracker: Creating new collection {collection_name}...")
        self._create_collection(collection_name)

        collection = COLLECTIONS_TYPES_MAP[collection_type](collection_name)
        self._open_collections.add(collection)
        print(f"QdrantTracker: New collection {collection_name} created and opened.")
        return collection
    
    def save_collection(self, collection_name, vector_size: int = 1536, embedding_model: str = "text-embedding-ada-002"):
        """
        Save the collection to Qdrant. Deletes and recreates the collection, then upserts all points.
        vector_size: must match the embedding model's output dimensions.
        embedding_model: OpenAI model used to embed chunks.
        """

        print(f"QdrantTracker: Deleting collection {collection_name}...")
        self._delete_collection(collection_name)
        print(f"QdrantTracker: Creating collection {collection_name}...")
        self._create_collection(collection_name, vector_size=vector_size)

        collection = self.get_collection(collection_name)
        points = collection.points_to_save(model_id=embedding_model)

        self._upsert_points(collection_name, points)
    
    def disconnect(self, collection_name):
        """
        Disconnect from the Qdrant collection.
        """
        self._check_open_collection(collection_name)
        
        while True:
            save_collection = input("Do you want to save the collection? (y/n): ")
            if save_collection.lower() == 'y':
                self.save_collection(collection_name)
                break
            elif save_collection.lower() == 'n':
                break
        

        self._remove(collection_name)
        print(f"QdrantTracker: Disconnected from collection: {collection_name}")
    
    def get_collection(self, collection_name):
        """
        Returns the collection object.
        """
        for c in self._open_collections:
            if c.get_collection_name() == collection_name:
                return c
        else:
            raise ValueError(f"Collection {collection_name} is not open.")

    def all_collections(self):
        collections = self._connection.get_collections().collections
        all_collections = [c.name for c in collections]
        all_collections.sort()
        return all_collections
    
    def open_collections(self):
        """
        Returns a list of currently open collections.
        """
        return [c.get_collection_name() for c in self._open_collections]
    
    def delete_collection(self, collection_name):
        if collection_name in [c.get_collection_name() for c in self._open_collections]:
            collection = self.get_collection(collection_name)
            self._open_collections.remove(collection)
        self._delete_collection(collection_name)

    def duplicate_collection(self, collection_name, new_collection_name):
        if self._existing_collection_name(new_collection_name):
            raise ValueError(f"Collection {new_collection_name} already exists.")
        self._connection.create_collection(
            collection_name=new_collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            init_from=models.InitFrom(collection=collection_name),
        )
        return

    """-----------------------------Private Methods-----------------------------"""
    def _remove(self, collection_name):
        self._check_open_collection(collection_name)
        collection = self.get_collection(collection_name)
        self._open_collections.remove(collection)

    def get_collection_type(self, point):
        """
        Get the type of collection from the point payload.
        """
        if "collection" in point:
            return point["collection"]["type"]
        else:
            raise ValueError("Point payload does not contain collection type information.")

    def _get_all_points_payload(self, collection_name, points=[]):
        offset = None
        while True:
            result, offset = self._connection.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            points.extend([point.payload for point in result])
            if offset is None:
                break #all points taken
        #print(f"Só para ver como estão os points: {points}")
        return points
    
    def _get_all_points(self, collection_name, points=[]):
        offset = None
        while True:
            result, offset = self._connection.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
            points.extend(result)
            if offset is None:
                break #all points taken
        #print(f"Só para ver como estão os points: {points}")
        return points

    def _make_qdrant_points(self, points):
        from qdrant_client.http.models import PointStruct

        qdrant_points = []

        for point in points:
            qdrant_points.append(PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload
            ))

        return qdrant_points

    def _upsert_points(self, collection_name, points):
        for i in range(0, len(points), 5):
            batch = points[i:i + 5]
            self._connection.upsert(
                collection_name=collection_name,
                wait = True,
                points=batch
            )

    def _check_open_collection(self, collection_name):
        if collection_name not in [c.get_collection_name() for c in self._open_collections]:
            raise ValueError(f"Collection {collection_name} is not open.")
        
    def _existing_collection_name(self, collection_name):
        collections = self._connection.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def _delete_collection(self, collection_name):
        self._connection.delete_collection(collection_name)  

    def _create_collection(self, collection_name, vector_size: int = 1536):
        # to create the collection if it does not exist
        self._connection.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def append_points_to_collection(self, collection_name: str, points: list):
        """Upsert points into an existing collection without deleting it first.
        Creates the collection if it doesn't exist. Used for multi-source ingestion."""
        if not self._existing_collection_name(collection_name):
            self._create_collection(collection_name)
            print(f"QdrantTracker: Created new collection '{collection_name}'.")
        print(f"QdrantTracker: Appending {len(points)} points to '{collection_name}'…")
        self._upsert_points(collection_name, points)
        print(f"QdrantTracker: Done appending to '{collection_name}'.")

    # -------------------------------------------------------------------------
    # Incremental sync — re-scrape URLs, compare content_hash, re-embed changes
    # -------------------------------------------------------------------------

    def sync_collection(self, scraper_name: str, collection_name: str, scraper_options: dict = None) -> dict:
        """
        Incrementally sync a URL-scraped collection:
          1. Re-scrape all URLs using the named scraper
          2. Compare content_hash per URL with what's in Qdrant
          3. Delete points for removed URLs
          4. Re-embed and insert changed or new pages
          5. Skip unchanged pages (hash matches)

        Returns a diff dict: {added, updated, deleted, unchanged, errors}
        """
        import hashlib
        from datetime import datetime, timezone
        from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, PointIdsList
        from ingestion.scrapers.runner import run_scraper
        from vectorization import get_embedding, get_point_id

        diff = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0, "errors": []}

        # 1. Re-scrape all URLs
        try:
            _, scraped_items = run_scraper(scraper_name, scraper_options or {})
        except Exception as e:
            diff["errors"].append(f"Scrape failed: {e}")
            return diff

        fresh_by_url = {item["url"]: item["text"] for item in scraped_items}
        if not fresh_by_url:
            diff["errors"].append("Scraper returned no items — aborting sync to avoid wiping collection.")
            return diff

        # 2. Get existing {source_url: content_hash} from Qdrant
        existing_by_url = self._get_existing_hashes_by_url(collection_name)

        # 3. Delete points for URLs no longer in sitemap
        stale_urls = set(existing_by_url.keys()) - set(fresh_by_url.keys())
        for url in stale_urls:
            try:
                self._delete_points_by_url(collection_name, url)
                diff["deleted"] += 1
            except Exception as e:
                diff["errors"].append(f"Delete failed for {url}: {e}")

        # 4. For each current URL: skip unchanged, replace changed, insert new
        now_iso = datetime.now(timezone.utc).isoformat()
        for url, fresh_text in fresh_by_url.items():
            fresh_hash = hashlib.sha256(fresh_text.encode("utf-8")).hexdigest()
            if url in existing_by_url:
                if existing_by_url[url] == fresh_hash:
                    diff["unchanged"] += 1
                    continue
                # Changed — delete old points then insert fresh
                try:
                    self._delete_points_by_url(collection_name, url)
                except Exception as e:
                    diff["errors"].append(f"Delete-before-update failed for {url}: {e}")
                    continue
                diff["updated"] += 1
            else:
                diff["added"] += 1

            # Embed and insert one point per page (structured scraping: 1 page = 1 chunk)
            try:
                vector = get_embedding(fresh_text)
                payload = {
                    "collection": {"type": "scs"},
                    "point": {
                        "text": fresh_text,
                        "source": scraper_name,
                        "source_url": url,
                        "content_hash": fresh_hash,
                        "scraped_at": now_iso,
                    },
                }
                self._upsert_points(collection_name, [PointStruct(
                    id=get_point_id(),
                    vector=vector,
                    payload=payload,
                )])
            except Exception as e:
                diff["errors"].append(f"Embed/insert failed for {url}: {e}")

        return diff

    def _get_existing_hashes_by_url(self, collection_name: str) -> dict:
        """
        Scroll all points that have a source_url, return {url: content_hash}.
        Uses Qdrant FieldCondition(is_empty=False) to filter only URL-annotated points.
        """
        from qdrant_client.http.models import Filter, FieldCondition

        scroll_filter = Filter(
            must=[FieldCondition(key="point.source_url", is_empty=False)]
        )
        existing = {}
        offset = None
        while True:
            result, offset = self._connection.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=100,
            )
            for point in result:
                p = point.payload.get("point", {})
                url = p.get("source_url")
                h = p.get("content_hash")
                if url:
                    existing[url] = h  # may be None for legacy points
            if offset is None:
                break
        return existing

    def _delete_points_by_url(self, collection_name: str, url: str) -> None:
        """
        Delete all Qdrant points whose point.source_url matches the given URL.
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointIdsList

        url_filter = Filter(
            must=[FieldCondition(key="point.source_url", match=MatchValue(value=url))]
        )
        # Collect IDs to delete via scroll (delete-by-filter not always available on older Qdrant)
        ids_to_delete = []
        offset = None
        while True:
            result, offset = self._connection.scroll(
                collection_name=collection_name,
                scroll_filter=url_filter,
                with_payload=False,
                with_vectors=False,
                offset=offset,
                limit=100,
            )
            ids_to_delete.extend(point.id for point in result)
            if offset is None:
                break

        if ids_to_delete:
            self._connection.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
            )

    def delete_points_by_ids(self, collection_name: str, point_ids: list) -> int:
        """
        Delete specific Qdrant points by their IDs.
        Returns the number of points requested for deletion.
        """
        from qdrant_client.http.models import PointIdsList
        if not point_ids or not self._existing_collection_name(collection_name):
            return 0
        self._connection.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        return len(point_ids)

    def scroll_points_by_urls(self, collection_name: str, urls: list) -> list:
        """
        Scroll all points whose point.source_url is in the given URL list.
        Returns list of Record objects (id + payload, no vector).
        Falls back to client-side filtering if Qdrant lacks a payload index.
        """
        if not urls or not self._existing_collection_name(collection_name):
            return []

        url_set = set(urls)

        # Try server-side filter first (fast, requires payload index)
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchAny
            scroll_filter = Filter(
                must=[FieldCondition(key="point.source_url", match=MatchAny(any=urls))]
            )
            records = []
            offset = None
            while True:
                result, offset = self._connection.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                    limit=100,
                )
                records.extend(result)
                if offset is None:
                    break
            return records
        except Exception:
            pass  # Fall back to client-side filtering

        # Fallback: scroll all and filter client-side
        records = []
        offset = None
        while True:
            result, offset = self._connection.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                offset=offset,
                limit=100,
            )
            for r in result:
                p = r.payload.get("point", {})
                if p.get("source_url") in url_set:
                    records.append(r)
            if offset is None:
                break
        return records

    def update_point(self, collection_name: str, point_id: str, new_text: str,
                     new_vector: list, original_text: str | None = None) -> bool:
        """
        Update a single point's text, vector, and mark it as manually edited.
        Preserves original_text on first edit; subsequent edits keep it unchanged.
        Returns True on success, False on failure.
        """
        import hashlib
        from datetime import datetime, timezone

        try:
            new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
            now_iso = datetime.now(timezone.utc).isoformat()

            payload_update = {
                "text": new_text,
                "content_hash": new_hash,
                "manually_edited": True,
                "edited_at": now_iso,
            }
            # Only set original_text on first edit
            if original_text is not None:
                payload_update["original_text"] = original_text

            self._connection.set_payload(
                collection_name=collection_name,
                payload=payload_update,
                points=[point_id],
                key="point",
            )
            self._connection.update_vectors(
                collection_name=collection_name,
                points=[
                    models.PointVectors(id=point_id, vector=new_vector)
                ],
            )
            return True
        except Exception as e:
            print(f"[QdrantTracker] update_point failed for {point_id}: {e}")
            return False

    def scroll_all(self, collection_name: str, limit: int = 200) -> list:
        """
        Return all point payloads from a collection (no query/filter).
        Used for FAQ table generation — retrieves all stored chunks.
        Returns a flat list of payload dicts (unwrapped from nested 'point' key if present).
        """
        if not self._existing_collection_name(collection_name):
            return []
        all_payloads = []
        offset = None
        while True:
            results, next_offset = self._connection.scroll(
                collection_name=collection_name,
                limit=min(limit, 100),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in results:
                p = r.payload or {}
                # Unwrap nested payload structure used by SCS_Collection
                all_payloads.append(p.get("point", p))
            if next_offset is None or len(all_payloads) >= limit:
                break
            offset = next_offset
        return all_payloads[:limit]
