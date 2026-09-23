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