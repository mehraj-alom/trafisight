import sqlite3
from config.global_config import GlobalConfig
class ViolationDB:

    def __init__(self,db_path=GlobalConfig.DATABASE_PATH):
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )
        self.cursor = self.conn.cursor()
        self.create_tables()


    def create_tables(self):

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                plate_number TEXT,

                violation_type TEXT,

                camera_id TEXT,

                track_id INTEGER,

                confidence REAL,

                timestamp TEXT,

                evidence_image TEXT,

                human_review_status TEXT

                reviewed_by TEXT

                review_timestamp TEXT

                status TEXT DEFAULT 'PENDING',

                risk_score TEXT,

                agent_score REAL,

                reasoning TEXT
            )
            """
        )
        self.conn.commit()



    def insert_violation(self,plate_number,violation_type,camera_id,track_id,confidence,timestamp,evidence_image):
        self.cursor.execute(
            """
            INSERT INTO violations (

                plate_number,
                violation_type,
                camera_id,
                track_id,
                confidence,
                timestamp,
                evidence_image

            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plate_number,
                violation_type,
                camera_id,
                track_id,
                confidence,
                timestamp,
                evidence_image
            )
        )
        self.conn.commit()


    def count_offences(self,plate_number):
        self.cursor.execute(
            """
            SELECT COUNT(*)
            FROM violations
            WHERE plate_number=?
            """,
            (plate_number,)
        )
        return self.cursor.fetchone()[0]
    

    def get_all_violations(self):
        self.cursor.execute(
            """
            SELECT *
            FROM violations
            """
        )

        return self.cursor.fetchall()
    

    def get_pending_violations(self):
        self.cursor.execute(
            """
            SELECT *
            FROM violations
            WHERE status='PENDING'
            """
        )
        return self.cursor.fetchall()
    

    def update_agent_result(self,violation_id,decision,risk_score,agent_score,reasoning):
        self.cursor.execute(
            """
            UPDATE violations
            SET
            status=?,
            risk_score=?,
            agent_score=?,
            reasoning=?
            WHERE id=?
            """,
            (
                decision,
                risk_score,
                agent_score,
                reasoning,
                violation_id
            )
        )
        self.conn.commit()