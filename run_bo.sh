for i in {1..100}; do
    python function.py --name $1   
    exit_status=$?

    
    if [ "$exit_status" -eq 1 ]; then
        echo "Python script failed with exit code: $exit_status"
        exit 1 # Exit with a non-zero status indicating failure
    fi
    python BO.py --name $1
    
    if [ "$exit_status" -eq 1 ]; then
        echo "Python script failed with exit code: $exit_status"
        exit 1 # Exit with a non-zero status indicating failure
    fi
done
