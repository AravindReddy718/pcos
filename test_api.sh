
echo "Testing Positive Case..."
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"follicle_r": 15, "follicle_l": 13, "skin_darkening": 1, "hair_growth": 1, "weight_gain": 1, "cycle_ri": 4}'

echo "\nTesting Negative Case..."
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"follicle_r": 2, "follicle_l": 2, "skin_darkening": 0, "hair_growth": 0, "weight_gain": 0, "cycle_ri": 2}'
