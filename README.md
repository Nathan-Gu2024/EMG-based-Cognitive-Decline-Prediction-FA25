// Single-banked RMW
// per-word clk-en, no per-bit mux, no 4608 equality tree, per-word (32 bit) SFR decode
logic [NIE-1:0] w_rm_clr, w_rm_set, w_cmm_clr, w_cmm_set;
logic [NIE-1:0] w_cmm_input_set_bm;
logic [NIE-1:0] w_rm_input_set_bm;
genvar wd;
generate 
	for (wd = 0; wd < NIE/AHB_HST_DW; wd++) begin
		assign w_rm_clr [32 * wd +:32] = {32{i_BitMapCtrl_RegIsRMClr_write_access[wd]}} & i_BitMapCtrl_RegIsRMClr_write_data[32 * wd +: 32];
		assign w_rm_set [32 * wd +:32] = {32{i_BitMapCtrl_RegIsRMSet_write_access[wd]}} & i_BitMapCtrl_RegIsRMSet_write_data[32 * wd +: 32];
		assign w_cmm_clr [32 * wd +:32] = {32{i_BitMapCtrl_RegIsCMMClr_write_access[wd]}} & i_BitMapCtrl_RegIsCMMClr_write_data[32 * wd +: 32];
		assign w_cmm_set [32 * wd +:32] = {32{i_BitMapCtrl_RegIsCMMSet_write_access[wd]}} & i_BitMapCtrl_RegIsCMMSet_write_data[32 * wd +: 32];
	end
endgenerate

// one-hot vecs via single barrel decoders

logic [NIE - 1 : 0] w_rel_clr;
assign w_rel_clr = o_cet_release_req ? (NIE'(1'b1) << o_cet_release_id) : 0;

logic [NIE - 1 : 0] w_pending_oh;
assign w_pending_oh = r_cet_release_pending_vld ? (NIE'(1'b1) << r_cet_release_pending) : 0;

// storing 144 word banked rmw for rm and cmm
genvar bw, si;
generate
	for (bw = 0; bw < NIE/AHB_HST_DW; bw++) begin : REL_BANK
		logic [31:0] rm_touched = w_rm_clr[bw * 32 +: 32] | w_rm_set[bw * 32 +: 32] | w_rel_clr[bw * 32 +: 32] | w_rm_input_set_bm[bw * 32 +: 32];
		logic [31:0] cmm_touched = w_cmm_clr[bw * 32 +: 32] | w_cmm_set[bw * 32 +: 32] | w_rel_clr[bw * 32 +: 32] | w_cmm_input_set_bm[bw * 32 +: 32];
		
		logic ce_rm = |rm_touched;
		logic ce_cmm = |cmm_touched;
		
		always_ff @(posedge i_clk, negedge i_rst_b) begin
			if (~i_rst_b) begin	
				r_rm_release_bitmap[32 * bw +: 32] <= 0;
			end else if (ce_rm) begin
				r_rm_release_bitmap[32 * bw +: 32] <= (r_rm_release_bitmap[bw * 32 +: 32] & ~rm_touched) // hold minus clears
									| (w_rm_set[bw * 32 +: 32] & ~w_rm_clr[bw * 32 +: 32]) // cpu set 
									| (w_rm_input_set_bm[bw * 32 +: 32] & ~rm_touched); // index set (lowest priority)
			end 
		end

		always_ff @(posedge i_clk, negedge i_rst_b) begin
			if (~i_rst_b) begin	
				r_cmm_release_bitmap[32 * bw +: 32] <= 0;
			end else if (ce_rm) begin
				r_cmm_release_bitmap[32 * bw +: 32] <= (r_cmm_release_bitmap[bw * 32 +: 32] & ~cmm_touched) // hold minus clears
									| (w_cmm_set[bw * 32 +: 32] & ~w_cmm_clr[bw * 32 +: 32]) // cpu set 
									| (w_cmm_input_set_bm[bw * 32 +: 32] & ~cmm_touched); // index set (lowest priority)
			end 
		end
	end
endgenerate


// read plan for rm and cmm and not pending (one hot inhibit and feeds all 
generate 
	for (si = 0; si < NIE; si++) begin
		assign w_cet_release_bitmap[si] = r_rm_release_bitmap[si] & r_cmm_release_bitmap[si] & ~w_pending_oh[si];
	end
endgenerate
		
