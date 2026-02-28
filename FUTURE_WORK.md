# Future Work

## Testing

### Python API Tests (pytest + TestClient)
- Test `/api/solutions` endpoint (list, create, update)
- Test `/api/solutions/{id}/collections` endpoint
- Test `/api/wizard/confirm` and `/api/wizard/save` / `/api/wizard/load`
- Test `/api/solutions/{id}/language` PUT
- Test `/api/solutions/add-collection` and `/api/solutions/delete-collection`
- Test `/api/workflow/step` for each pipeline step
- Test `/api/settings` GET/PUT

### Browser / UI Tests (Playwright)
- Global solution selector: pick solution, switch tabs, verify it persists
- Global solution selector: "+ Create new solution" flow
- Global solution language badge: display and edit
- Work with RAG: collection pills render for selected solution
- Chat tab: collections load when solution is selected
- Analyse Site: button naming ("Select Analyse Mode", "Select", "Launch Analyse Site")
- Analyse Site: saved session modal flow (mode selection, launch)
- Analyse Site → Work with RAG: navigate to specific collection via `_openCollectionInBuildRag()`
- Recent files: click switches global solution if file belongs to different solution
