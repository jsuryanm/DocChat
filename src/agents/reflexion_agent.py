class ReflexionAgent:

    def decide(self,
               grounded,
               quality,
               retries,
               max_retries):
        
        if grounded and quality == "HIGH":
            return "accept"
        
        if retries >= max_retries:
            return "stop"
        
        return "retry"