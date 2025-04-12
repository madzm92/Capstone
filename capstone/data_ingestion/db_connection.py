from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


#DB Set Up
class DatabaseConnection:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = None
    
    def connect(self):
        self.session = self.Session()
        return self.session
    
    def close(self):
        if self.session:
            self.session.close()
        
    def __enter__(self):
        return self.connect()
    
    def __exit__(self):
        self.close()