"""Tool library management for end mills and v-bits."""

import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ToolType(Enum):
    ENDMILL = "endmill"
    VBIT = "vbit"


@dataclass
class EndMill:
    """Flat end mill for roughing/clearing."""
    id: str
    name: str
    diameter: float  # mm
    feed_rate: float  # mm/min
    plunge_rate: float  # mm/min
    spindle_rpm: int
    flute_count: int = 2
    stepover_percent: float = 40.0  # percentage of diameter
    max_depth_per_pass: float = 2.0  # mm
    tool_type: ToolType = field(default=ToolType.ENDMILL)
    
    @property
    def stepover(self) -> float:
        """Calculate stepover distance in mm."""
        return self.diameter * (self.stepover_percent / 100.0)
    


@dataclass
class VBit:
    """V-bit for v-carving."""
    id: str
    name: str
    diameter: float  # mm
    feed_rate: float  # mm/min
    plunge_rate: float  # mm/min
    spindle_rpm: int
    angle: float = 60.0  # included angle in degrees
    tip_diameter: float = 0.0  # flat tip diameter, 0 for sharp
    tool_type: ToolType = field(default=ToolType.VBIT)
    
    def depth_for_width(self, width: float) -> float:
        """Calculate cutting depth needed for a given carved width."""
        import math
        half_angle = math.radians(self.angle / 2)
        effective_width = width - self.tip_diameter
        if effective_width <= 0:
            return 0.0
        return (effective_width / 2) / math.tan(half_angle)


# Type alias for any tool
Tool = EndMill | VBit


class ToolLibrary:
    """SQLite-based persistent storage for tools."""
    
    DEFAULT_ENDMILLS = [
        ("em-6mm-2f", "6mm 2-Flute End Mill", 6.0, 1000.0, 300.0, 12000, 2, 40.0, 2.0),
        ("em-3mm-2f", "3mm 2-Flute End Mill", 3.0, 800.0, 200.0, 15000, 2, 40.0, 1.5),
    ]
    
    DEFAULT_VBITS = [
        ("vb-60deg", "60° V-Bit", 12.0, 600.0, 150.0, 12000, 60.0, 0.0),
        ("vb-90deg", "90° V-Bit", 12.0, 500.0, 120.0, 10000, 90.0, 0.0),
    ]
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            config_dir = Path.home() / ".config" / "zcarve"
            config_dir.mkdir(parents=True, exist_ok=True)
            db_path = config_dir / "tools.db"
        
        self.db_path = db_path
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _init_db(self) -> None:
        """Initialize database schema and default tools."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Create endmills table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS endmills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    diameter REAL NOT NULL,
                    feed_rate REAL NOT NULL,
                    plunge_rate REAL NOT NULL,
                    spindle_rpm INTEGER NOT NULL,
                    flute_count INTEGER DEFAULT 2,
                    stepover_percent REAL DEFAULT 40.0,
                    max_depth_per_pass REAL DEFAULT 2.0
                )
            """)
            
            # Create vbits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vbits (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    diameter REAL NOT NULL,
                    feed_rate REAL NOT NULL,
                    plunge_rate REAL NOT NULL,
                    spindle_rpm INTEGER NOT NULL,
                    angle REAL DEFAULT 60.0,
                    tip_diameter REAL DEFAULT 0.0
                )
            """)
            
            # Insert defaults if tables are empty
            cursor.execute("SELECT COUNT(*) FROM endmills")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("""
                    INSERT INTO endmills 
                    (id, name, diameter, feed_rate, plunge_rate, spindle_rpm, flute_count, stepover_percent, max_depth_per_pass)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, self.DEFAULT_ENDMILLS)
            
            cursor.execute("SELECT COUNT(*) FROM vbits")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("""
                    INSERT INTO vbits 
                    (id, name, diameter, feed_rate, plunge_rate, spindle_rpm, angle, tip_diameter)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, self.DEFAULT_VBITS)
            
            conn.commit()
    
    def add(self, tool: Tool) -> None:
        """Add or update a tool."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            if isinstance(tool, EndMill):
                cursor.execute("""
                    INSERT OR REPLACE INTO endmills 
                    (id, name, diameter, feed_rate, plunge_rate, spindle_rpm, flute_count, stepover_percent, max_depth_per_pass)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (tool.id, tool.name, tool.diameter, tool.feed_rate, tool.plunge_rate,
                      tool.spindle_rpm, tool.flute_count, tool.stepover_percent, tool.max_depth_per_pass))
            elif isinstance(tool, VBit):
                cursor.execute("""
                    INSERT OR REPLACE INTO vbits 
                    (id, name, diameter, feed_rate, plunge_rate, spindle_rpm, angle, tip_diameter)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (tool.id, tool.name, tool.diameter, tool.feed_rate, tool.plunge_rate,
                      tool.spindle_rpm, tool.angle, tool.tip_diameter))
            
            conn.commit()
    
    def remove(self, tool_id: str) -> None:
        """Remove a tool by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM endmills WHERE id = ?", (tool_id,))
            cursor.execute("DELETE FROM vbits WHERE id = ?", (tool_id,))
            conn.commit()
    
    def get(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Check endmills
            cursor.execute("SELECT * FROM endmills WHERE id = ?", (tool_id,))
            row = cursor.fetchone()
            if row:
                return EndMill(
                    id=row[0], name=row[1], diameter=row[2], feed_rate=row[3],
                    plunge_rate=row[4], spindle_rpm=row[5], flute_count=row[6],
                    stepover_percent=row[7], max_depth_per_pass=row[8]
                )
            
            # Check vbits
            cursor.execute("SELECT * FROM vbits WHERE id = ?", (tool_id,))
            row = cursor.fetchone()
            if row:
                return VBit(
                    id=row[0], name=row[1], diameter=row[2], feed_rate=row[3],
                    plunge_rate=row[4], spindle_rpm=row[5], angle=row[6],
                    tip_diameter=row[7]
                )
            
            return None
    
    def get_endmills(self) -> list[EndMill]:
        """Get all end mills."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM endmills ORDER BY name")
            return [
                EndMill(
                    id=row[0], name=row[1], diameter=row[2], feed_rate=row[3],
                    plunge_rate=row[4], spindle_rpm=row[5], flute_count=row[6],
                    stepover_percent=row[7], max_depth_per_pass=row[8]
                )
                for row in cursor.fetchall()
            ]
    
    def get_vbits(self) -> list[VBit]:
        """Get all v-bits."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vbits ORDER BY name")
            return [
                VBit(
                    id=row[0], name=row[1], diameter=row[2], feed_rate=row[3],
                    plunge_rate=row[4], spindle_rpm=row[5], angle=row[6],
                    tip_diameter=row[7]
                )
                for row in cursor.fetchall()
            ]
    
    def get_all(self) -> list[Tool]:
        """Get all tools."""
        return self.get_endmills() + self.get_vbits()
