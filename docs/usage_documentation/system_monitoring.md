# System Monitoring

## Check overall system health

To verify the API is up and reachable:

```
health_check()
```

To get detailed workload information (active jobs, queue pending count, workload level):

```
get_system_status()
```

The same data is available as a resource:

```
virtual-streamer://system/status
```

---

## Get the stream configuration

To retrieve the stream's settings and details:

```
get_stream_config()
```

Combined stream config and active programmation are available as a resource:

```
virtual-streamer://stream/config
```
