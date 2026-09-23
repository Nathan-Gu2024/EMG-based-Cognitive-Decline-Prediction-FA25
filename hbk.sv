always_comb begin
	pos = 0; 
	for (i = 0; i < NIE/32; i++)
		if (i_BitMapCtrl_RegIsCMMClr_write_access[i]) begin
			pos = i * 32; 
		end 
end

generate 
	for (i = 0; i < 32; i = i + 1) begin
		always_ff @ (posedge i_clk or negedge i_rst_b) begin
			if (~i_rst_b)
				r_cmm_release_bitmap[i] <= 1'h0 ;
			else if (i_BitMapCtrl_RegIsCMMClr_write_access[NIE/32 - 1:0] & i_BitMapCtrl_RegIsCMMClr_write_data[pos+i])
				r_cmm_release_bitmap[pos+i] <= 1'b0 ;
			else if (i_BitMapCtrl_RegIsCMMSet_write_access[NIE/32 - 1:0] & i_BitMapCtrl_RegIsCMMSet_write_data[pos+i])
				r_cmm_release_bitmap[pos+i] <= 1'b1 ;
			else if (o_cet_release_req & i == 0)
				r_cmm_release_bitmap[o_cet_release_id] <= 1'b0 ;
			else if (i_cmm_crb_cet_index_vld & i == 0)
				r_cmm_release_bitmap[i_cmm_crb_cet_index] <= 1'b1 ; 
		end

endgenerate

always_comb begin
	pos = 0; 
	for (i = 0; i < NIE/32; i++)
		if (i_BitMapCtrl_RegIsRMClr_write_access[i]) begin
			pos = i * 32; 
		end 
end

generate 
	for (i = 0; i < 32; i = i + 1) begin
		always_ff @ (posedge i_clk or negedge i_rst_b) begin
			if (~i_rst_b)
				r_rm_release_bitmap[i] <= 1'h0 ;
			else if (i_BitMapCtrl_RegIsRMClr_write_access[NIE/32 - 1:0] & i_BitMapCtrl_RegIsRMClr_write_data[pos+i])
				r_rm_release_bitmap[pos+i] <= 1'b0 ;
			else if (i_BitMapCtrl_RegIsRMSet_write_access[NIE/32 - 1:0] & i_BitMapCtrl_RegIsRMSet_write_data[pos+i])
				r_rm_release_bitmap[pos+i] <= 1'b1 ;
			else if (o_cet_release_req & i == 0)
				r_rm_release_bitmap[o_cet_release_id] <= 1'b0 ;
			else if (i_rm_crb_cet_index_vld & i == 0)
				r_rm_release_bitmap[i_rm_crb_cet_index] <= 1'b1 ; 
		end

endgenerate
