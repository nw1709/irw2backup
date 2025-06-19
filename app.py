# --- Consensus System ---
def achieve_consensus_multi(ocr_text):
    """Consensus zwischen Claude und GPT"""
    
    # Lösungen generieren
    with st.spinner("Claude löst..."):
        claude_solution = solve_with_claude(ocr_text)
    
    with st.spinner(f"{GPT_MODEL} löst..."):
        gpt_solution = solve_with_gpt(ocr_text)
    
    # Antworten extrahieren
    claude_answers = extract_answers(claude_solution)
    gpt_answers = extract_answers(gpt_solution)
    
    # Debug Info
    with st.expander("🔍 Debug: Antworten"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Claude:**")
            st.json(claude_answers)
        with col2:
            st.write(f"**{GPT_MODEL}:**")
            st.json(gpt_answers)
    
    # Vergleiche
    all_tasks = set(claude_answers.keys()) | set(gpt_answers.keys())
    consensus = True
    
    for task in sorted(all_tasks):
        claude_ans = claude_answers.get(task, "?")
        gpt_ans = gpt_answers.get(task, "?")
        
        if claude_ans != gpt_ans:
            consensus = False
            st.warning(f"Diskrepanz bei {task}: Claude={claude_ans}, GPT={gpt_ans}")
    
    if consensus and claude_answers:
        st.success("✅ Modelle sind sich einig!")
        return claude_solution
    else:
        st.info("🔄 Verwende Claude mit Verifikation")
        # Selbst-Verifikation
        verify_prompt = (
            f"Prüfe diese Lösung kritisch:\n\n"
            f"AUFGABE:\n{ocr_text}\n\n"
            f"LÖSUNG:\n{claude_solution}\n\n"
            f"Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β!\n\n"
            f"Gib die FINALE KORREKTE Lösung im gleichen Format."
        )

        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": verify_prompt}]
        )
        
        return response.content[0].text
