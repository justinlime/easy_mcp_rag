from .config import Config
from .server import MCPRAGServer

def main():
    config = Config.from_args()
    server = MCPRAGServer(config)
    
    try:
        server.initialize()
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
