from supply_chain_model import *
import matplotlib.pyplot as plt

# NVIDIA sold 500,000 H100 GPUs in Q3 2023
# https://www.tomshardware.com/tech-industry/nvidia-ai-and-hpc-gpu-sales-reportedly-approached-half-a-million-units-in-q3-thanks-to-meta-facebook
# xAI setups 100,000 H100 GPUs
# https://www.tomshardware.com/pc-components/gpus/elon-musk-took-19-days-to-set-up-100-000-nvidia-h200-gpus-process-normally-takes-4-years

# inflation from 2016 to 2024 = 31%
# https://www.in2013dollars.com/us/inflation/2016?amount=1
inflation_factor = 1.31

pn_i = 10                   # 7nm processing node
CENT_frontend_eng_weeks = 52
CENT_frontend_num_engineers = 25
CENT_backend_eng_weeks = 52
CENT_backend_num_engineers = 25

Mask_Cost = mw_mask_costs_arr[pn_i] * inflation_factor
Backend_Labor_Cost = mw_backend_labor_cost_per_week * CENT_backend_num_engineers * CENT_backend_eng_weeks * inflation_factor
Backend_CAD_Cost = mw_backend_cad_license_per_eng_week * CENT_backend_eng_weeks * inflation_factor
Frontend_Labor_Cost = mw_frontend_labor_cost_per_week * CENT_frontend_num_engineers * CENT_frontend_eng_weeks * inflation_factor
IP_Licensing_Cost = mw_ip_costs_arr[pn_i] * inflation_factor
Packaging_Cost = mw_flip_chip_bga_package_design_cost * inflation_factor
System_Cost = mw_system_NRE_cost * inflation_factor
Total_NRE_Cost = Mask_Cost + Backend_Labor_Cost + Backend_CAD_Cost + Frontend_Labor_Cost + IP_Licensing_Cost + Packaging_Cost + System_Cost
print(f"Mask Cost: ${Mask_Cost:,.2f}")
print(f"Backend Labor Cost: ${Backend_Labor_Cost:,.2f}")
print(f"Backend CAD Cost: ${Backend_CAD_Cost:,.2f}")
print(f"Frontend Labor Cost: ${Frontend_Labor_Cost:,.2f}")
print(f"IP Licensing Cost: ${IP_Licensing_Cost:,.2f}")
print(f"Packaging Cost: ${Packaging_Cost:,.2f}")
print(f"System Cost: ${System_Cost:,.2f}")
print(f"Total NRE Cost: ${Total_NRE_Cost:,.2f}")

# Considering other AI provider and cloud provider, 5x xAI's volume
CENT_product_volume = 376000 * 8
CENT_PNM_die_size = 1.02
CENT_DRAM_controller_die_size = 4.5
CENT_DRAM_PHY_size = 10.8
CENT_PHIe_controller_PHY_size = 2.64
CENT_die_size = (CENT_PNM_die_size + CENT_DRAM_controller_die_size + CENT_DRAM_PHY_size + CENT_PHIe_controller_PHY_size)
CENT_yield_rate = yield_rate(CENT_die_size, defect_density_vector_mm2[pn_i], 3)
CENT_num_wafers_needed = num_wafers_needed(CENT_die_size, CENT_product_volume)
CENT_die_cost = wafer_costs_arr[pn_i] * CENT_num_wafers_needed / CENT_product_volume / CENT_yield_rate

# https://semiengineering.com/using-machine-learning-to-increase-yield-and-lower-packaging-costs/
# 2D packing accounts for 29% of Chip cost
CENT_package_cost = CENT_die_cost * 0.29/(1-0.29)
CENT_device_cost = CENT_die_cost + CENT_package_cost
print(f"CENT die cost: ${CENT_die_cost:,.2f}")
print(f"CENT package cost: ${CENT_package_cost:,.2f}")

Device_NRE_cost = Total_NRE_Cost / CENT_product_volume
Device_cost = Device_NRE_cost + CENT_device_cost
print(f"CENT Device Non-NRE Cost: ${Device_NRE_cost:,.2f}")
print(f"CENT Device cost: ${Device_cost:,.2f}")
import matplotlib.pyplot as plt

# Define the costs
CENT_costs = {
    "die cost": CENT_die_cost,
    "package cost": CENT_package_cost,
    "NRE_costs": Device_NRE_cost
}

NRE_costs = {
    "Mask Cost": Mask_Cost,
    "Backend Labor Cost": Backend_Labor_Cost,
    "Backend CAD Cost": Backend_CAD_Cost,
    "Frontend Labor Cost": Frontend_Labor_Cost,
    "IP Licensing Cost": IP_Licensing_Cost,
    "Packaging Cost": Packaging_Cost,
    "System Cost": System_Cost
}

# # Plotting
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 2))

# # Colors for each cost component
# colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightblue', 'lightgray']

# # Initialize the bottom position for stacking
# bottom = 0

# # Plot each NRE cost component as a stacked bar
# for (label, cost), color in zip(NRE_costs.items(), colors):
#     ax1.barh('NRE Cost Components', cost, left=bottom, color=color, label=label)
#     bottom += cost

# # Colors for CENT cost components
# cent_colors = ['lightcyan', 'lightyellow', 'lightgoldenrodyellow']

# # Initialize the bottom position for stacking
# bottom = 0

# # Plot each CENT cost component as a stacked bar
# for (label, cost), color in zip(CENT_costs.items(), cent_colors):
#     ax2.barh('CENT Cost Components', cost, left=bottom, color=color, label=label)
#     bottom += cost

# # Add labels and title
# ax1.set_xlabel('Cost Value')
# ax1.set_title('NRE Cost Components')
# ax1.legend(loc='upper right')

# ax2.set_xlabel('Cost Value')
# ax2.set_title('CENT Cost Components')
# ax2.legend(loc='upper right')

# # Adjust layout and save the plot as a PDF file
# plt.tight_layout()
# plt.savefig('cost_components_percentage.pdf', format='pdf')

TPU_frontend_eng_weeks = 104
TPU_frontend_num_engineers = 100
TPU_backend_eng_weeks = 52
TPU_backend_num_engineers = 40

Mask_Cost = mw_mask_costs_arr[pn_i] * inflation_factor
Backend_Labor_Cost = mw_backend_labor_cost_per_week * TPU_backend_num_engineers * TPU_backend_eng_weeks * inflation_factor
Backend_CAD_Cost = mw_backend_cad_license_per_eng_week * TPU_backend_eng_weeks * inflation_factor
Frontend_Labor_Cost = mw_frontend_labor_cost_per_week * TPU_frontend_num_engineers * TPU_frontend_eng_weeks * inflation_factor
IP_Licensing_Cost = mw_ip_costs_arr[pn_i] * inflation_factor
Packaging_Cost = mw_flip_chip_bga_package_design_cost * inflation_factor
System_Cost = mw_system_NRE_cost * inflation_factor
Total_NRE_Cost = Mask_Cost + Backend_Labor_Cost + Backend_CAD_Cost + Frontend_Labor_Cost + IP_Licensing_Cost + Packaging_Cost + System_Cost

TPU_product_volume = 376000 * 8 / 12
TPU_die_size = 600
TPU_yield_rate = yield_rate(TPU_die_size, defect_density_vector_mm2[pn_i], 3)
TPU_num_wafers_needed = num_wafers_needed(TPU_die_size, TPU_product_volume)
TPU_die_cost = wafer_costs_arr[pn_i] * TPU_num_wafers_needed / TPU_product_volume / TPU_yield_rate

# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6962745
# 144mm2 2.5D packaging cost is $16.93, considering 33% inflation from 2014 to 2024, $22.52
# https://www.anandtech.com/show/9969/jedec-publishes-hbm2-specification
# HBM2 die size 92mm2
TPU_package_cost = (TPU_die_size + 4 * 92) / 144 * 22.52
TPU_device_cost = TPU_die_cost + TPU_package_cost
print(f"NeuPIM die cost: ${TPU_die_cost:,.2f}")
print(f"NeuPIM package cost: ${TPU_package_cost:,.2f}")
Device_NRE_cost = Total_NRE_Cost / TPU_product_volume
Device_cost = Device_NRE_cost + TPU_device_cost
print(f"NeuPIM Device Non-NRE Cost: ${Device_NRE_cost:,.2f}")
print(f"NeuPIM Device cost: ${Device_cost:,.2f}")
