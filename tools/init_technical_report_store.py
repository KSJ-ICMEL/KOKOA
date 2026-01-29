"""
Technical Report Store Manager
==============================
Manage the technical report vector store.

Usage:
  python manage_report_store.py             # View all reports
  python manage_report_store.py --force     # Delete ALL reports
  python manage_report_store.py --select    # Interactively select and delete reports
"""

import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from kokoa.config import Config


def get_store_path():
    return os.path.join(Config.INITIAL_STATE_DIR, "technical_report_store")


def list_reports():
    """List all reports in the store"""
    store_path = get_store_path()
    if not os.path.exists(store_path):
        print("⚠️ Technical report store does not exist yet.")
        return []
    
    try:
        client = chromadb.PersistentClient(path=store_path)
        collection = client.get_collection("technical_reports")
        results = collection.get(include=["documents", "metadatas"])
        
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        ids = results.get("ids", [])
        
        reports = []
        for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
            reports.append({
                "index": i + 1,
                "id": doc_id,
                "content": doc[:100] + "..." if len(doc) > 100 else doc,
                "result_type": meta.get("result_type", "?"),
                "target": meta.get("target", "?")[:50],
                "run_id": meta.get("run_id", "?")
            })
        return reports
    except Exception as e:
        print(f"❌ Error reading store: {e}")
        return []


def display_reports(reports):
    """Display reports in a numbered list"""
    if not reports:
        print("📭 No reports found.")
        return
    
    print(f"\n📊 Technical Reports ({len(reports)} total)")
    print("=" * 80)
    for r in reports:
        status = "✅" if r["result_type"] == "SUCCESS" else "❌" if r["result_type"] == "FAILURE" else "⚪"
        print(f"{r['index']:3}. {status} [{r['result_type']:8}] {r['target']}")
        print(f"     Run: {r['run_id']}")
    print("=" * 80)


def force_reset():
    """Delete entire store"""
    store_path = get_store_path()
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        print(f"🗑️ Deleted: {store_path}")
    else:
        print("⚠️ Store does not exist.")
    print("✅ Technical report store reset complete.")


def select_and_delete():
    """Interactive selection and deletion"""
    reports = list_reports()
    if not reports:
        return
    
    display_reports(reports)
    
    print("\nEnter report numbers to delete (e.g., '1 3 5' or 'all' or 'cancel'):")
    user_input = input("> ").strip().lower()
    
    if user_input == "cancel":
        print("❌ Cancelled.")
        return
    
    if user_input == "all":
        confirm = input("⚠️ Delete ALL reports? (yes/no): ").strip().lower()
        if confirm == "yes":
            force_reset()
        else:
            print("❌ Cancelled.")
        return
    
    # Parse numbers
    try:
        indices = [int(x) for x in user_input.split()]
    except ValueError:
        print("❌ Invalid input. Enter numbers separated by spaces.")
        return
    
    # Get IDs to delete
    ids_to_delete = []
    for idx in indices:
        for r in reports:
            if r["index"] == idx:
                ids_to_delete.append(r["id"])
                break
    
    if not ids_to_delete:
        print("❌ No valid reports selected.")
        return
    
    # Delete from ChromaDB
    store_path = get_store_path()
    client = chromadb.PersistentClient(path=store_path)
    collection = client.get_collection("technical_reports")
    collection.delete(ids=ids_to_delete)
    
    print(f"✅ Deleted {len(ids_to_delete)} report(s).")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage technical report store")
    parser.add_argument("--force", action="store_true", help="Delete ALL reports")
    parser.add_argument("--select", action="store_true", help="Interactively select and delete")
    args = parser.parse_args()
    
    print("[Technical Report Store Manager]")
    print("=" * 40)
    print(f"Store: {get_store_path()}")
    
    if args.force:
        confirm = input("⚠️ This will DELETE ALL reports. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            force_reset()
        else:
            print("❌ Cancelled.")
    elif args.select:
        select_and_delete()
    else:
        # Default: just list
        reports = list_reports()
        display_reports(reports)


if __name__ == "__main__":
    main()
