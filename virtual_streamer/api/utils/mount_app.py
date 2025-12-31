"""
Mountable App Utilities.

Provides utilities for mounting sub-applications (like ADK agents) onto the main
FastAPI application with support for:
- Lifespan merging (sub-app startup/shutdown runs with main app)
- OpenAPI docs merging (all routes appear in unified /docs)
- Route conflict detection
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List, Self

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, model_validator
from starlette.applications import Starlette
from starlette.routing import Mount, Route

import logging

logger = logging.getLogger(__name__)

# Default documentation routes to exclude when merging docs
DEFAULT_DOCUMENTATION_ROUTES = [
    "openapi",
    "swagger_ui_html",
    "swagger_ui_redirect",
    "redoc_html",
]


class MountableApp(BaseModel):
    """Configuration for a mountable sub-application.
    
    Attributes:
        name: Identifier for the mounted app
        app: The ASGI application to mount (FastAPI, Starlette, or any ASGI app)
        path: URL path prefix where the app will be mounted (e.g., "/adk")
        merge_lifespan: Whether to merge startup/shutdown events with main app
        protected: Whether the mounted app requires authentication (placeholder)
        merge_docs: Whether to merge OpenAPI documentation with main app
        exclude_routes_names: Route names to exclude from the mounted app
    """

    name: str = Field(..., description="The name of the mounted app.")
    app: Starlette = Field(..., description="The mounted application.")
    path: str = Field(..., description="The path of the mounted app.")
    merge_lifespan: bool = Field(
        default=True,
        description="Whether to merge the lifespan of the main app and the mounted app.",
    )
    protected: bool = Field(
        default=False,
        description="Whether the mounted app is protected (placeholder for auth).",
    )
    merge_docs: bool = Field(
        default=True,
        description="Whether to merge the OpenAPI documentation of the main app and the mounted app.",
    )
    exclude_routes_names: List[str] = Field(
        default_factory=list,
        description="The names of the routes to exclude from the mounted app.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def fill_exclude_routes_names_if_merge_docs(self) -> Self:
        """Auto-exclude documentation routes when merging docs."""
        if self.merge_docs:
            self.exclude_routes_names.extend(DEFAULT_DOCUMENTATION_ROUTES)
        return self


def mount_app(
    main_app: Starlette | FastAPI,
    mounted_app: MountableApp,
) -> None:
    """Mount a sub-application onto the main application.

    This function mounts a sub-application onto a main FastAPI/Starlette application
    with support for:
    - Filtering out specific routes (like duplicate /docs)
    - Merging lifespan events (startup/shutdown)
    - Merging OpenAPI documentation (for FastAPI apps)

    Args:
        main_app: The main FastAPI or Starlette application
        mounted_app: Configuration for the app to mount

    Example:
        ```python
        from fastapi import FastAPI
        from virtual_streamer.api.utils.mount_app import mount_app, MountableApp

        main_app = FastAPI()
        sub_app = FastAPI()

        mounted = MountableApp(
            app=sub_app,
            path="/subapi",
            name="subapi",
            merge_lifespan=True,
            merge_docs=True,
        )
        mount_app(main_app, mounted)
        ```
    """
    app_to_mount = mounted_app.app

    # Filter out excluded routes
    app_to_mount = _filter_routes(app_to_mount, mounted_app.exclude_routes_names)

    # Merge lifespans so sub-app startup/shutdown runs with main app
    if mounted_app.merge_lifespan:
        _merge_lifespan(main_app, app_to_mount)

    # Merge OpenAPI documentation for unified /docs
    if isinstance(main_app, FastAPI) and mounted_app.merge_docs:
        _merge_docs(main_app, mounted_app)

    # Mount the app at the specified path
    main_app.mount(mounted_app.path, app_to_mount, name=mounted_app.name)
    
    logger.info(f"Mounted app '{mounted_app.name}' at path '{mounted_app.path}'")


def _merge_lifespan(main_app: Starlette, mounted_app: Starlette) -> None:
    """Merge the lifespan of the main app and the mounted app.

    This wraps the main app's lifespan context manager to also include
    the mounted app's lifespan, ensuring proper startup/shutdown ordering.

    Args:
        main_app: The main application
        mounted_app: The mounted application
    """
    # Store the original lifespan context to avoid infinite recursion
    original_lifespan = main_app.router.lifespan_context

    @asynccontextmanager
    async def merged_lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        async with original_lifespan(app):
            # Wrap the lifespan of the mounted app in the lifespan of the main app
            async with mounted_app.router.lifespan_context(app):
                yield

    main_app.router.lifespan_context = merged_lifespan


def _filter_routes(app: Starlette, exclude_routes_names: List[str]) -> Starlette:
    """Filter out routes from the mounted app.

    Args:
        app: The mounted application
        exclude_routes_names: The names of the routes to exclude

    Returns:
        The app with excluded routes removed
    """
    routes_to_remove = [
        route
        for route in app.routes
        if getattr(route, "name", None) in exclude_routes_names
    ]
    for route in routes_to_remove:
        app.routes.remove(route)
    return app


def _get_mount_openapi(mount_app: Mount, prefix_path: str) -> List[dict[str, Any]]:
    """Get the OpenAPI schema of a mounted app.

    A Mount object can host any kind of ASGI app, including FastAPI and Starlette.
    For custom ASGI apps, we only add a path to the OpenAPI schema.

    Args:
        mount_app: The mounted application
        prefix_path: The prefix path of the mounted app

    Returns:
        List of OpenAPI schema dictionaries
    """
    if isinstance(mount_app.app, (Starlette, FastAPI)):
        return _get_app_openapi(
            mount_app.app, f"{prefix_path.removesuffix('/')}{mount_app.path}"
        )
    else:
        # For custom ASGI apps, only add a path to the OpenAPI schema
        return [
            {
                "paths": {
                    f"{prefix_path.removesuffix('/')}{mount_app.path}": {
                        "post": {
                            "description": "Possible downstream method for an arbitrary ASGI mounted app."
                        },
                    }
                }
            }
        ]


def _get_route_openapi(route: Route, prefix_path: str) -> dict[str, Any]:
    """Get the OpenAPI schema of a route.

    Args:
        route: The route
        prefix_path: The prefix path of the route

    Returns:
        OpenAPI schema dictionary
    """
    if isinstance(route.endpoint, Starlette):
        return get_openapi(
            title="to-be-replaced",
            version="to-be-replaced",
            routes=[
                APIRoute(
                    path=f"{prefix_path.removesuffix('/')}{route.path}",
                    endpoint=route.endpoint,
                    methods=route.methods,
                    name=route.name,
                )
            ],
        )
    # For custom ASGI apps, only add a path to the OpenAPI schema
    return {
        "paths": {
            f"{prefix_path.removesuffix('/')}{route.path}": {
                "post": {
                    "description": "Possible downstream method for an arbitrary ASGI mounted app."
                },
            }
        }
    }


def _get_app_openapi(app: Starlette | FastAPI, prefix_path: str) -> List[dict[str, Any]]:
    """Get the OpenAPI schema of an app.

    Args:
        app: The app
        prefix_path: The prefix path of the app

    Returns:
        List of OpenAPI schema dictionaries
    """
    if isinstance(app, FastAPI):
        # Override path prefix for FastAPI app
        openapi_schema = app.openapi()
        openapi_schema["paths"] = {
            f"{prefix_path.removesuffix('/')}{path}": schema
            for path, schema in openapi_schema["paths"].items()
        }
        return [openapi_schema]

    sub_openapi_list: List[dict[str, Any]] = []
    for route in app.routes:
        if isinstance(route, Route):
            sub_openapi_list.append(_get_route_openapi(route, prefix_path))
        elif isinstance(route, Mount):
            sub_openapi_list.extend(_get_mount_openapi(route, prefix_path))
        else:
            # Websocket and Host from Starlette are not supported
            logger.warning(f"Merging API docs not supported for route type {type(route)}")
    return sub_openapi_list


def _merge_docs(main_app: FastAPI, mounted_app: MountableApp) -> FastAPI:
    """Merge the OpenAPI schema of the main app and the mounted app.

    Args:
        main_app: The main FastAPI application
        mounted_app: The mounted application configuration

    Returns:
        The main app with merged OpenAPI schema
    """
    # Extract routes from the mounted app and add them to the main app OpenAPI schema
    sub_openapi_list = _get_app_openapi(mounted_app.app, mounted_app.path)

    def _merge_openapi(
        main_openapi: dict[str, Any], sub_openapi: dict[str, Any]
    ) -> dict[str, Any]:
        main_openapi.get("paths", {}).update(sub_openapi.get("paths", {}))
        main_openapi.get("components", {}).get("securitySchemes", {}).update(
            sub_openapi.get("components", {}).get("securitySchemes", {})
        )
        main_openapi.get("components", {}).get("schemas", {}).update(
            sub_openapi.get("components", {}).get("schemas", {})
        )
        main_openapi.get("tags", []).extend(sub_openapi.get("tags", []))
        return main_openapi

    updated_openapi = main_app.openapi()
    for sub_openapi in sub_openapi_list:
        updated_openapi = _merge_openapi(updated_openapi, sub_openapi)

    main_app.openapi_schema = updated_openapi
    return main_app


def check_route_conflicts(
    main_app: FastAPI | Starlette, mounted_apps: List[MountableApp]
) -> None:
    """Check for route conflicts between main app and mounted apps.

    This function analyzes all routes in the main application and mounted
    applications to detect potential conflicts where multiple routes serve
    the same path with overlapping HTTP methods.

    Args:
        main_app: The main application
        mounted_apps: List of apps to be mounted

    Raises:
        ValueError: If route conflicts are detected
    """
    # Check that no mounted app has the same path
    paths = [mounted_app.path for mounted_app in mounted_apps or []]
    if len(set(paths)) != len(paths):
        raise ValueError("Multiple mounted apps serve the same path")

    # Dictionary to track all routes: {(path, method): [(app_name, route_info), ...]}
    route_registry: dict[tuple[str, str], List[tuple[str, dict[str, Any]]]] = {}

    def register_route(
        path: str, methods: set[str], app_name: str, route_name: str | None = None
    ) -> None:
        """Register a route and its methods."""
        for method in methods:
            key = (path, method)
            route_info = {
                "name": route_name or "unnamed",
                "path": path,
                "method": method,
            }
            if key not in route_registry:
                route_registry[key] = []
            route_registry[key].append((app_name, route_info))

    def extract_routes_from_app(
        app: FastAPI | Starlette,
        app_name: str,
        path_prefix: str = "",
        excluded_route_names: List[str] | None = None,
    ) -> None:
        """Extract routes from an app and register them."""
        for route in app.routes:
            if isinstance(route, (Route, APIRoute)):
                # Construct full path with prefix
                full_path = (
                    f"{path_prefix.rstrip('/')}{route.path}"
                    if path_prefix and route.path != "/"
                    else route.path
                )
                if path_prefix and route.path == "/":
                    full_path = path_prefix

                methods: set[str] = getattr(route, "methods", set()) or set()
                route_name = getattr(route, "name", None)

                if route_name not in (excluded_route_names or []):
                    register_route(full_path, methods, app_name, route_name)

            elif isinstance(route, Mount):
                # Handle nested mounts
                if isinstance(route.app, (FastAPI, Starlette)):
                    nested_prefix = f"{path_prefix.rstrip('/')}{route.path}"
                    extract_routes_from_app(
                        route.app,
                        f"{app_name} -> {route.name or 'unnamed_mount'}",
                        nested_prefix,
                    )
                else:
                    # For custom ASGI apps, register as wildcard
                    full_path = (
                        f"{path_prefix.rstrip('/')}{route.path}"
                        if path_prefix
                        else route.path
                    )
                    mount_name = f"{app_name} -> {route.name or 'unnamed_asgi_mount'}"
                    wildcard_methods = {
                        "GET",
                        "POST",
                        "PUT",
                        "DELETE",
                        "PATCH",
                        "HEAD",
                        "OPTIONS",
                    }
                    register_route(
                        full_path,
                        wildcard_methods,
                        mount_name,
                        f"asgi_app_{route.name or 'unnamed'}",
                    )
            else:
                logger.warning(f"Unsupported route type: {type(route)}")

    # Extract routes from main app
    extract_routes_from_app(main_app, "main_app")

    # Extract routes from mounted apps
    for mounted_app in mounted_apps:
        app_name = f"mounted_app[{mounted_app.name}]"
        extract_routes_from_app(
            mounted_app.app,
            app_name,
            path_prefix=mounted_app.path,
            excluded_route_names=mounted_app.exclude_routes_names,
        )

    # Check for conflicts
    conflicts = []
    for key, registrations in route_registry.items():
        path, method = key
        if len(registrations) > 1:
            conflict_info: dict[str, Any] = {
                "path": path,
                "method": method,
                "conflicting_routes": [],
            }
            for app_name, route_info in registrations:
                conflict_info["conflicting_routes"].append(
                    {
                        "app": app_name,
                        "route_name": route_info["name"],
                    }
                )
            conflicts.append(conflict_info)

    # Raise error if conflicts found
    if conflicts:
        error_parts = ["Route conflicts detected:"]
        for conflict in conflicts:
            path = conflict["path"]
            method = conflict["method"]
            route_details: List[str] = []
            for route in conflict["conflicting_routes"]:
                route_details.append(f"'{route['route_name']}' in {route['app']}")
            error_parts.append(f"  - {method} {path}: {', '.join(route_details)}")
        error_message = "\n".join(error_parts)
        raise ValueError(error_message)

