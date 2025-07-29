from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    title: str
    isDone: bool = False
    
    
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

items = []

@app.post("/items/")
async def create_item(item: Item):
    items.append(item)
    return {"message": "Item created successfully", "item": item}

@app.get("/items/", response_model=list[Item])
async def read_items():
    return {"items": items}

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    if item_id < 0 or item_id >= len(items):
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id < 0 or item_id >= len(items):
        return {"error": "Item not found"}
    deleted_item = items.pop(item_id)
    return {"message": "Item deleted successfully", "item": deleted_item}