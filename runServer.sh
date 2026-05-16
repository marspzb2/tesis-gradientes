#!/bin/bash
jupyter  notebook \
 --no-browser \
--allow-root \
--ip=0.0.0.0 \
--NotebookApp.shutdown_no_activity_timeout=16000