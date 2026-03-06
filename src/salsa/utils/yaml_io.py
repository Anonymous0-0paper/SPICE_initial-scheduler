import json
import os.path
from pathlib import Path

import yaml
import jinja2
from kubernetes import client
from kubernetes.client import ApiException

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

def to_yaml(data):
    if isinstance(data, tuple):
        data = list(data)
    return yaml.safe_dump(data, sort_keys=False)

def load_yamls(*filenames, context=None):
    templates = os.path.join(SRC_DIR, os.path.join('salsa', 'templates'))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(templates))
    env.filters.update({'to_yaml': to_yaml})

    rendered = []
    for name in filenames:
        template = env.get_template(name)
        rendered.append(template.render(context or {}))

    return tuple(rendered)

def load_manifests(*filenames, context=None):
    rendered = load_yamls(*filenames, context=context)

    manifests = []
    for yaml_str in rendered:
        manifests.append(yaml.safe_load(yaml_str))

    return tuple(manifests)

def verify_manifests(k8s_client, manifest_string, dry_run=True):
    try:
        docs = list(yaml.safe_load_all(manifest_string))
    except yaml.YAMLError as exc:
        print(f"YAML Syntax Error: {exc}")
        return False

    custom_api = client.CustomObjectsApi(k8s_client)
    dry_run_arg = "All" if dry_run else None

    for doc in docs:
        if not doc: continue

        group, version = doc['apiVersion'].split('/')
        kind = doc['kind']
        name = doc['metadata']['name'] # Extract name for patching

        plural_map = {
            'ServiceExport': 'serviceexports',
            'ServiceImport': 'serviceimports',
            'PropagationPolicy': 'propagationpolicies',
            'ClusterPropagationPolicy': 'clusterpropagationpolicies'
        }
        plural = plural_map.get(kind, kind.lower() + 's')

        try:
            # 1. Try to CREATE
            if doc["kind"].startswith("Cluster"):
                print(f"Creating Cluster Entity: {kind}/{name}...")
                custom_api.create_cluster_custom_object(
                    group=group,
                    version=version,
                    plural=plural,
                    body=doc,
                    dry_run=dry_run_arg
                )
            else:
                print(f"Creating Namespace Entity: {kind}/{name}...")
                namespace = doc['metadata'].get('namespace', 'default')
                custom_api.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=doc,
                    dry_run=dry_run_arg
                )
            print(" -> Created Successfully")

        except ApiException as e:
            # 2. If it fails with 409 (Conflict), try to PATCH (Update)
            if e.status == 409:
                print(f" -> Resource exists. Attempting to UPDATE (Patch)...")
                try:
                    if doc["kind"].startswith("Cluster"):
                        custom_api.patch_cluster_custom_object(
                            group=group,
                            version=version,
                            plural=plural,
                            name=name,  # Patch requires 'name' argument explicitly
                            body=doc,
                            dry_run=dry_run_arg
                        )
                    else:
                        namespace = doc['metadata'].get('namespace', 'default')
                        custom_api.patch_namespaced_custom_object(
                            group=group,
                            version=version,
                            namespace=namespace,
                            plural=plural,
                            name=name, # Patch requires 'name' argument explicitly
                            body=doc,
                            dry_run=dry_run_arg
                        )
                    print(" -> Updated Successfully")
                except ApiException as e_patch:
                    # If Patch also fails, then we actually report failure
                    print(f"FAILED TO UPDATE: {e_patch.reason}")
                    return False
            else:
                # 3. If it failed for any other reason (e.g., 400 Bad Request), print error
                try:
                    details = json.loads(e.body)
                    message = details.get("message", e.reason)
                    print(f"FAILED: {message}")
                except Exception:
                    print(f"FAILED: {e.body or e.reason}")
                return False

    return True