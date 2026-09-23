// original
logic [$clog2(NIE/32)-1:0] rm_pos;
logic [31:0]               rm_set_d, rm_clr_d;
always_comb begin
    rm_pos = '0; rm_set_d = '0; rm_clr_d = '0;
    for (int w = 0; w < NIE/32; w++) begin
        if (i_BitMapCtrl_RegIsRMSet_write_access[w] | i_BitMapCtrl_RegIsRMClr_write_access[w]) rm_pos = w;
        rm_set_d |= {32{i_BitMapCtrl_RegIsRMSet_write_access[w]}} & i_BitMapCtrl_RegIsRMSet_write_data[w*32 +: 32];
        rm_clr_d |= {32{i_BitMapCtrl_RegIsRMClr_write_access[w]}} & i_BitMapCtrl_RegIsRMClr_write_data[w*32 +: 32];
    end
end
always_ff @(posedge i_clk or negedge i_rst_b)
    if (~i_rst_b) r_rm_release_bitmap <= '0;
    else begin                                   // later assignment wins
        if (i_rm_crb_cet_index_vld) r_rm_release_bitmap[i_rm_crb_cet_index] <= 1'b1;
        if (o_cet_release_req)      r_rm_release_bitmap[o_cet_release_id]   <= 1'b0;
        for (int b = 0; b < 32; b++) begin
            if (rm_set_d[b]) r_rm_release_bitmap[rm_pos*32 + b] <= 1'b1;
            if (rm_clr_d[b]) r_rm_release_bitmap[rm_pos*32 + b] <= 1'b0;
        end
    end

// hybrid with per-word strobes, top-version ordering, single always_ff per vec, per-word ce
localparam int N_WORD = NIE/32;                 // 144

// ---- per-word strobe planes (wires; word-level CE; no 4608-bit barrels) ----
logic [N_WORD-1:0] rel_in_word, rm_idx_in, cmm_idx_in, rm_wr_clr, rm_wr_set, cmm_wr_clr, cmm_wr_set;
genvar w;
generate for (w = 0; w < N_WORD; w = w + 1) begin : REL_DEC
    assign rel_in_word [w] = o_cet_release_req         && (o_cet_release_id      [12:5] == w[7:0]);
    assign rm_idx_in   [w] = i_rm_crb_cet_index_vld    && (i_rm_crb_cet_index   [12:5] == w[7:0]);
    assign cmm_idx_in  [w] = i_cmm_crb_cet_index_vld   && (i_cmm_crb_cet_index [12:5] == w[7:0]);
    assign rm_wr_clr   [w] = i_BitMapCtrl_RegIsRMClr_write_access [w];
    assign rm_wr_set   [w] = i_BitMapCtrl_RegIsRMSet_write_access [w];
    assign cmm_wr_clr  [w] = i_BitMapCtrl_RegIsCMMClr_write_access[w];
    assign cmm_wr_set  [w] = i_BitMapCtrl_RegIsCMMSet_write_access[w];
end endgenerate

// ---- RM bitmap: one always_ff per vector, per-word CE, priority by order ----
always_ff @(posedge i_clk or negedge i_rst_b) begin
    if (~i_rst_b)
        r_rm_release_bitmap <= '0;
    else
        for (int w = 0; w < N_WORD; w++) begin
            if (rm_wr_set[w] | rm_wr_clr[w] | rel_in_word[w] | rm_idx_in[w]) begin   // word CE (strobes)
                // later assignment wins:  CLR > SET > release > index   (== reference)
                if (rm_idx_in  [w]) r_rm_release_bitmap[w*32 + i_rm_crb_cet_index  [4:0]] <= 1'b1;
                if (rel_in_word[w]) r_rm_release_bitmap[w*32 + o_cet_release_id    [4:0]] <= 1'b0;
                if (rm_wr_set  [w]) r_rm_release_bitmap[w*32 +: 32] <=
                        r_rm_release_bitmap[w*32 +: 32] | i_BitMapCtrl_RegIsRMSet_write_data[w*32 +: 32];
                if (rm_wr_clr  [w]) r_rm_release_bitmap[w*32 +: 32] <=
                        r_rm_release_bitmap[w*32 +: 32] & ~i_BitMapCtrl_RegIsRMClr_write_data[w*32 +: 32];
            end
        end
end

// ---- CMM bitmap: same, with cmm_* planes ----
always_ff @(posedge i_clk or negedge i_rst_b) begin
    if (~i_rst_b)
        r_cmm_release_bitmap <= '0;
    else
        for (int w = 0; w < N_WORD; w++) begin
            if (cmm_wr_set[w] | cmm_wr_clr[w] | rel_in_word[w] | cmm_idx_in[w]) begin
                if (cmm_idx_in [w]) r_cmm_release_bitmap[w*32 + i_cmm_crb_cet_index [4:0]] <= 1'b1;
                if (rel_in_word[w]) r_cmm_release_bitmap[w*32 + o_cet_release_id   [4:0]] <= 1'b0;
                if (cmm_wr_set [w]) r_cmm_release_bitmap[w*32 +: 32] <=
                        r_cmm_release_bitmap[w*32 +: 32] | i_BitMapCtrl_RegIsCMMSet_write_data[w*32 +: 32];
                if (cmm_wr_clr [w]) r_cmm_release_bitmap[w*32 +: 32] <=
                        r_cmm_release_bitmap[w*32 +: 32] & ~i_BitMapCtrl_RegIsCMMClr_write_data[w*32 +: 32];
            end
        end
end

// ---- one-hot pending inhibit + read plane (shared by A and B) ----
logic [NIE-1:0] w_pending_oh;
assign w_pending_oh = r_cet_release_pending_vld ? (NIE'(1'b1) << r_cet_release_pending) : '0;
genvar si;
generate for (si = 0; si < NIE; si = si + 1) begin : REL_PLANE
    assign w_cet_release_bitmap[si] = r_rm_release_bitmap[si] & r_cmm_release_bitmap[si] & ~w_pending_oh[si];
end endgenerate


// corrected version of the previous suggested design
// ---- per-direction word decode (no shared pos) + global activity CE ----
logic [$clog2(NIE/32)-1:0] rm_set_pos, rm_clr_pos, cmm_set_pos, cmm_clr_pos;
logic [31:0] rmset_d, rmclr_d, cmmset_d, cmmclr_d;
always_comb begin
    rm_set_pos = '0; rm_clr_pos = '0; cmm_set_pos = '0; cmm_clr_pos = '0;
    rmset_d = '0; rmclr_d = '0; cmmset_d = '0; cmmclr_d = '0;
    for (int w = 0; w < NIE/32; w++) begin
        if (i_BitMapCtrl_RegIsRMSet_write_access[w])  begin rm_set_pos = w;  rmset_d  = i_BitMapCtrl_RegIsRMSet_write_data [w*32 +: 32]; end
        if (i_BitMapCtrl_RegIsRMClr_write_access[w])  begin rm_clr_pos = w;  rmclr_d  = i_BitMapCtrl_RegIsRMClr_write_data [w*32 +: 32]; end
        if (i_BitMapCtrl_RegIsCMMSet_write_access[w]) begin cmm_set_pos = w; cmmset_d = i_BitMapCtrl_RegIsCMMSet_write_data[w*32 +: 32]; end
        if (i_BitMapCtrl_RegIsCMMClr_write_access[w]) begin cmm_clr_pos = w; cmmclr_d = i_BitMapCtrl_RegIsCMMClr_write_data[w*32 +: 32]; end
    end
end

// Priority-by-ordering ("later assignment wins"): CLR > SET > release > index  (== reference)
always_ff @(posedge i_clk or negedge i_rst_b) begin
    if (~i_rst_b)
        r_rm_release_bitmap <= '0;
    else if (i_rm_crb_cet_index_vld | o_cet_release_req | |rmset_d | |rmclr_d) begin
        if (i_rm_crb_cet_index_vld) r_rm_release_bitmap[i_rm_crb_cet_index] <= 1'b1;
        if (o_cet_release_req)      r_rm_release_bitmap[o_cet_release_id]  <= 1'b0;
        for (int b = 0; b < 32; b++) begin
            if (rmset_d[b]) r_rm_release_bitmap[rm_set_pos*32 + b] <= 1'b1;
            if (rmclr_d[b]) r_rm_release_bitmap[rm_clr_pos*32 + b] <= 1'b0;
        end
    end
end

always_ff @(posedge i_clk or negedge i_rst_b) begin
    if (~i_rst_b)
        r_cmm_release_bitmap <= '0;
    else if (i_cmm_crb_cet_index_vld | o_cet_release_req | |cmmset_d | |cmmclr_d) begin
        if (i_cmm_crb_cet_index_vld) r_cmm_release_bitmap[i_cmm_crb_cet_index] <= 1'b1;
        if (o_cet_release_req)       r_cmm_release_bitmap[o_cet_release_id]   <= 1'b0;
        for (int b = 0; b < 32; b++) begin
            if (cmmset_d[b]) r_cmm_release_bitmap[cmm_set_pos*32 + b] <= 1'b1;
            if (cmmclr_d[b]) r_cmm_release_bitmap[cmm_clr_pos*32 + b] <= 1'b0;
        end
    end
end

// Shared rm_pos between SET and CLR. rm_pos is written whenever either CSR write_access[w] is high and overwritten to the last matching word. If RegIsRMSet and RegIsRMClr strobe different words in one cycle, the data gathered for one word is applied at the other's address (mis-addressed clear/set). Reference per-bit code has no such coupling. Fix: split rm_set_pos/rm_clr_pos (each direction's own pointer) — every collision class then lands at the correct word. (Under AHB single-beat, at most one word per direction per cycle anyway; splitting makes even that moot.)
// No clock-enable (user review point 3). Fix: a global activity enable idx_vld | rel_req | |set_d | |clr_d around the edge block.
// Everything else in the top version is verified correct: full 4608×2 storage, and the priority-by-ordering idiom (order index(1st) → release(2nd) → SET(3rd) → CLR(4th), "later assignment wins") reproduces the reference per-bit priority clr > set > release > index on every same-bit collision pair (index∧set, index∧clr, index∧rel, set∧clr, set∧rel, rel∧clr) — walked and confirmed.


