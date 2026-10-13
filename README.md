# Interhack BCN 2026 · Estrella Damm Logistics Optimisation

A delivery-planning tool for a beverage distribution operation. It groups customer orders into vehicle-sized clusters, builds the daily routes for each vehicle, and plans how every truck is loaded at the warehouse.

Built for the Estrella Damm challenge at Interhack BCN 2026 by a team of four.

## The problem

Every day a fleet of trucks and vans leaves a single depot (DDI Mollet del Vallès) to deliver beer, water, soft drinks, wine, dairy, coffee and packaging to hundreds of customers. Each customer has its own time window, products differ in weight and fragility, and trucks have to be loaded in the right order. The goal is to reduce travel time, fuel, CO₂ and the number of trips, while making the warehouse's job easier.

The dataset covers five delivery days (16–20 March 2026), with a mixed fleet of 8-pallet trucks, 6-pallet trucks and 3-pallet vans.

## How it works
orders + time windows + product data
       
logistics distance matrix 
        │
        ▼
constrained K-Medoids clustering  ──►  
 
refinement (move / swap)
      │
      ▼
vehicle assignment per cluster ──►  warehouse zones (A–H)
      │
      ▼
per-cluster routing (nearest neighbour + 2-opt) with time windows
      │
      ▼
warehouse pallet planning (LIFO, fragility, returns)
  
1. Routing by logistics clustering (clustering/, vrp/)
Logistics distance. Orders are compared with a weighted cost instead of plain geography: D = α·travel time + β·fuel + γ·distance proxy + δ·logistics penalty (weights 0.5 / 0.3 / 0.1 / 0.1, tunable in config.py).
Constrained K-Medoids (PAM-style) with K-Medoids++ seeding, one cluster per vehicle. Clusters that exceed the largest vehicle's pallet capacity are split afterwards, then improved with move/swap refinement passes.
Vehicle assignment. Small, priority, non-fragile clusters go to the van; the rest go to 6- or 8-pallet trucks, balancing trips across the fleet.
Routing inside each cluster. Nearest-neighbour construction + 2-opt improvement. The cost model includes time windows (penalties scale with how tight the window is), unloading time proportional to pallets, simulated parking time, depot reloads and an 8-hour daily cap that is enforced after routing (overflow stops are redistributed when possible).
Baseline. run_week.py solves the same days with Clarke–Wright savings plus simulated annealing (2-opt, relocate and swap moves) on a combined time + fuel cost, for comparison.
Emissions. Fuel is converted to CO₂ with a diesel factor of 2.68 kg/L.

2. Warehouse preparation (warehouse/, run_warehouse.py)
Each product is classified by category, fragility, weight, stackability and returnability.
Products are aggregated by reference across all stops of a trip and bin-packed into pallets, heavy items first, with 15 % of each pallet reserved for next-day returns.
Pallets follow a LIFO loading order (last delivery goes deepest in the truck) and are tagged with a loading position.
Minor categories (< 8 % of an order) are routed to a mixed zone; each trip is assigned one of eight warehouse zones.
The cost model compares preparation strategies (by order, by reference, hybrid) and penalises fragile/heavy mixes, stacking violations, and fragile goods buried deep in vans without side access.

3. Interfaces (static HTML)

Web views that read the generated JSON results:

View	Purpose
clusters.html, zones.html	Cluster map and warehouse zones
week.html	Weekly view of routes
driver.html	Per-vehicle route sheet for drivers
warehouse.html, warehouse_worker.html	Pallet plan and picking checklist for warehouse staff
frontend.html	Entry point
Project structure
clustering/        K-Medoids, logistics distance, refinement, TSP/2-opt, warehouse zones, cost
vrp/               Order/vehicle models, Clarke-Wright savings, simulated annealing, cost
warehouse/         Product classifier and warehouse cost model
data/              Loaders and cached travel-time / fuel matrices
config.py          All tunable parameters (fleet, times, penalties, weights)
run_clustering.py  Clustering pipeline  ->  results_clustering.json
run_warehouse.py   Pallet planning      ->  results_warehouse.json
run_driver_data.py Driver route sheets  ->  results_driver.json
run_week.py        Savings + SA baseline ->  results_week.json
*.html             Visualisation views
Getting started
bash
pip install -r requirements.txt

# optional: only needed to rebuild travel matrices; without it, a haversine x 35 km/h proxy is used
echo "GMAPS_API_KEY=your_key" > .env

python run_clustering.py                 # all five days (or --date DD/MM/YYYY)
python run_warehouse.py
python run_driver_data.py

python -m http.server 8000               # then open http://localhost:8000/frontend.html

Input data. The challenge spreadsheets (Hackaton.xlsx, Horarios Entrega.XLSX, ZM040.XLSX) were provided by the organisers and are not included in this repository. config.py expects them in ~/Downloads/Hackaton.

Notes and limitations
The solvers are heuristics built within a hackathon timeframe; they do not guarantee optimal solutions.
Penalty weights and handling times are modelling assumptions, not measured values.
The "CO₂ saved" figure in the driver data is an illustrative fixed-ratio estimate, not a measured saving.
There are no automated tests.
Team

Built by a team of four. Igor Bolige focused on the warehouse and truck-loading module (pallet planning and the warehouse interfaces), and all members contributed across the pipeline. The code was developed with AI-assisted tooling.
