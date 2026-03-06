from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict, Optional, List
import requests
from torchgen.packaged.autograd.gen_trace_type import emit_trace_body

from salsa.utils.typing import application_id, microservice_id


class ThanosQuery:
    def __init__(self, config: Dict):
        self.thanos_url = config["thanos"]["url"]
        self.thanos_queries: Dict[str, str] = config["thanos"]["queries"]

    def run_query(self, query_name: str) -> Optional[Dict[application_id, Dict[microservice_id, float]]]:
        base_query = self.thanos_queries.get(query_name)
        if not base_query:
            print(f"Unknown query name: {query_name}")
            return None

        try:
            response = requests.get(
                f"{self.thanos_url}/api/v1/query",
                params={'query': base_query},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Thanos Connection Error: {e}")
            return None

        results = data.get('data', {}).get('result', [])

        query_result = {}
        for result in results:
            metric = result.get('metric', {})
            if metric:
                svc_name = metric.get('app_name', 'InvalidService')
                application_name = metric.get('application_name', 'InvalidApplication')
            else:
                svc_name = 'InvalidService'
                application_name = 'InvalidApplication'

            if application_name not in query_result:
                query_result[application_name] = {}
            query_result[application_name][svc_name] = float(result['value'][1])

        return query_result

    def run_all_queries(self, query_names: List[str]) -> Dict[str, Optional[Dict[application_id, Dict[microservice_id, float]]]]:
        metrics = {}

        with ThreadPoolExecutor(max_workers=len(query_names)) as executor:
            future_to_query = {
                executor.submit(self.run_query, q_name): q_name
                for q_name in query_names
            }
            for future in as_completed(future_to_query):
                query_name = future_to_query[future]
                try:
                    metrics[query_name] = future.result()
                except Exception as exc:
                    print(f"{query_name} generated an exception: {exc}")
                    metrics[query_name] = None

        return metrics
