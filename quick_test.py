import requests

categories = ['age', 'civil_status', 'destination', 'education', 'occupation', 'sex', 'origin']

print("\n" + "="*70)
print("TESTING ALL CATEGORIES - 3 YEAR PREDICTIONS")
print("="*70)

for cat in categories:
    response = requests.post(
        f"http://localhost:5432/api/predict/{cat}",
        json={"years_ahead": 3} # Change years_ahead to test different prediction horizons
    )
    
    if response.status_code == 200:
        data = response.json()
        predictions = data['predictions']
        
        # Calculate total for first year
        first_year = predictions[0]
        total = sum(first_year['predictions'].values())
        
        print(f"\n✅ {cat.upper():15s} | Year {first_year['year']}: {total:10,.0f} total emigrants")
        
        # Show Top 3 categories
        sorted_preds = sorted(
            first_year['predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for rank, (category, value) in enumerate(sorted_preds, 1):
            print(f"   {rank}. {category:25s}: {value:8,.0f}")
    else:
        print(f"❌ {cat.upper():15s} | Error: {response.json()['error']}")

print("\n" + "="*70)