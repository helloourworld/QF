def calculate_cnq_fair_value():
    # --- INPUTS (POST-SPLIT 2026 DATA) ---
    # Normalized FCF is more realistic for "Fair Value" than "Current FCF"
    current_fcf = 8.75      # Billions CAD (2025 Actual FCF)
    normalized_fcf = 7.5      # Billions CAD (Assuming long-term oil at $70-75)
    growth_rate = 0.02        # 2% (Standard for mature oil & gas)
    discount_rate = 0.095     # 9.5% (WACC: Energy companies have higher risk)
    terminal_growth = 0.02    # 2% (Inflation/Terminal growth)
    net_debt = 9.4            # Billions CAD 
    shares_outstanding = 2.13 # Billions (THE CORRECT POST-SPLIT COUNT)

    # 1. Project FCF for 5 years and discount them
    discounted_fcf_sum = 0
    current_fcf = normalized_fcf
    
    for year in range(1, 6):
        current_fcf *= (1 + growth_rate)
        pv_fcf = current_fcf / (1 + discount_rate)**year
        discounted_fcf_sum += pv_fcf
        
    # 2. Terminal Value (Gordon Growth Method)
    # FCF in Year 6
    fcf_year_6 = current_fcf * (1 + terminal_growth)
    terminal_value = fcf_year_6 / (discount_rate - terminal_growth)
    
    # Discount Terminal Value to Year 0
    pv_terminal_value = terminal_value / (1 + discount_rate)**5
    
    # 3. Calculate Enterprise Value & Equity Value
    enterprise_value = discounted_fcf_sum + pv_terminal_value
    equity_value = enterprise_value - net_debt
    
    fair_value_per_share = equity_value / shares_outstanding
    
    print("-" * 30)
    print(f"Projected EV:       ${enterprise_value:.2f} B")
    print(f"Equity Value:       ${equity_value:.2f} B")
    print(f"Fair Value / Share: ${fair_value_per_share:.2f} CAD")
    print("-" * 30)

calculate_cnq_fair_value()


"""
Total Debt: ~$17.15 Billion CAD
Cash: ~$0.5 - $1.0 Billion CAD (CNQ keeps very little cash because they use it all to pay dividends and buy back shares).
Net Debt: ~$16.5 Billion CAD.
Note: CNQ has a policy where once they hit a certain Net Debt target (recently $10B), they return 100% of their Free Cash Flow to shareholders.
"""

"""
Crucial Note: Total Liabilities is not the same as Total Debt. Liabilities include things like "Accounts Payable" (money owed to suppliers) which doesn't charge interest.

Total Debt
Definition: Specifically the money the company has borrowed from banks or bondholders to fund operations.
The Formula:
Total Debt = Short-Term Debt + Long-Term Debt

(Short Term Debt + Long Term Debt) - Cash = Net Debt


CNQ views $10 billion as the "Safe Floor." Once they reached that floor, they decided that the shareholders deserved the cash more than the banks did. As long as oil is above $40/barrel, CNQ can easily manage $10B in debt while making you rich through dividends.


"""