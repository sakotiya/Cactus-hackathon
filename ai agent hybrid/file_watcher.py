"""
File Watcher - Monitor memory folder and auto-reload
"""

import os
import time
import hashlib
import threading
from typing import Set, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyPDF2 import PdfReader
from docx import Document


class MemoryFileHandler(FileSystemEventHandler):
    """Handle file system events in memory folder"""
    
    def __init__(self, callback: Callable):
        self.callback = callback
        self.processed_files: Set[str] = set()
    
    def on_created(self, event):
        if not event.is_directory:
            self.callback(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory:
            self.callback(event.src_path)


class FileWatcher:
    """Monitor memory folder and process new files"""
    
    def __init__(self, memory_folder: str, process_callback: Callable):
        """
        Initialize file watcher
        
        Args:
            memory_folder: Path to memory folder
            process_callback: Function to call when file is added/modified
                             Should accept (filepath, content, file_hash)
        """
        self.memory_folder = memory_folder
        self.process_callback = process_callback
        self.processed_hashes: Set[str] = set()
        self.observer = None
        self.running = False
        
        # Create folder if doesn't exist
        os.makedirs(memory_folder, exist_ok=True)
    
    def _get_file_hash(self, filepath: str) -> str:
        """Calculate file hash to detect changes"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _read_file(self, filepath: str) -> str:
        """Read file content based on extension"""
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext == '.pdf':
                return self._read_pdf(filepath)
            elif ext == '.docx':
                return self._read_docx(filepath)
            elif ext in ['.txt', '.md', '.markdown']:
                return self._read_text(filepath)
            else:
                return ""
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
    
    def _read_pdf(self, filepath: str) -> str:
        """Read PDF file"""
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except:
            return ""
    
    def _read_docx(self, filepath: str) -> str:
        """Read DOCX file"""
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except:
            return ""
    
    def _read_text(self, filepath: str) -> str:
        """Read text file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except:
            return ""
    
    def _process_file(self, filepath: str) -> None:
        """Process a single file"""
        # Skip hidden files and non-supported formats
        filename = os.path.basename(filepath)
        if filename.startswith('.'):
            return
        
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ['.pdf', '.docx', '.txt', '.md', '.markdown']:
            return
        
        # Calculate hash
        file_hash = self._get_file_hash(filepath)
        if not file_hash or file_hash in self.processed_hashes:
            return  # Already processed
        
        print(f"📄 Processing new file: {filename}")
        
        # Read content
        content = self._read_file(filepath)
        if not content:
            print(f"   ⚠️  Could not read content from {filename}")
            return
        
        # Call callback
        try:
            self.process_callback(filepath, content, file_hash)
            self.processed_hashes.add(file_hash)
            print(f"   ✅ Processed {filename}")
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    def scan_existing_files(self) -> None:
        """Scan and process existing files in folder"""
        print(f"🔍 Scanning {self.memory_folder} for existing files...")
        
        for filename in os.listdir(self.memory_folder):
            filepath = os.path.join(self.memory_folder, filename)
            if os.path.isfile(filepath):
                self._process_file(filepath)
    
    def start(self, scan_existing: bool = True) -> None:
        """Start watching the folder"""
        if self.running:
            return
        
        print(f"👁️  Starting file watcher on {self.memory_folder}")
        
        # Scan existing files first
        if scan_existing:
            self.scan_existing_files()
        
        # Start watchdog observer
        event_handler = MemoryFileHandler(self._process_file)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.memory_folder, recursive=False)
        self.observer.start()
        self.running = True
        
        print(f"✅ File watcher active - monitoring every 30 seconds")
    
    def stop(self) -> None:
        """Stop watching"""
        if self.observer and self.running:
            self.observer.stop()
            self.observer.join()
            self.running = False
            print("🛑 File watcher stopped")
    
    def start_background(self, scan_existing: bool = True) -> threading.Thread:
        """Start watcher in background thread"""
        thread = threading.Thread(target=self.start, args=(scan_existing,), daemon=True)
        thread.start()
        return thread
