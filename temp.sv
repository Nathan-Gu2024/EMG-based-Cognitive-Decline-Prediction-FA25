module nvme_crb_cda_logic
import nvme_crb_pkg::*;
(
    
i_clk,
i_rst_b,

i_rm_crb_cet_index,
i_rm_crb_cet_index_vld,
o_rm_crb_rdy,

i_cmm_crb_cet_index,
i_cmm_crb_cet_index_vld,
o_cmm_crb_rdy,

//QM Interface
o_qm_cnt_dec_vld,
o_qm_cnt_dec_data,
i_qm_cnt_dec_rdy,

//ACIQ Push Interface
o_sfa_cda_aciq_wvld,
o_sfa_cda_aciq_wdata,
i_sfa_cda_aciq_status,
i_sfa_cda_aciq_wrdy,

//WLD TIME OUT
i_wld_timeout_vld,
i_wld_timeout_cet_index,

i_wld_release_cet_vld,
o_wld_release_cet_rdy,

//FW Interface for RM Release BITMAP

i_BitMapCtrl_RegIsRMClr_write_access,
i_BitMapCtrl_RegIsRMSet_write_access,
i_BitMapCtrl_RegIsRMClr_write_data,
i_BitMapCtrl_RegIsRMSet_write_data,

//FW Interface for CMM Release BITAMP
i_BitMapCtrl_RegIsCMMClr_write_access,
i_BitMapCtrl_RegIsCMMSet_write_access,
i_BitMapCtrl_RegIsCMMClr_write_data,
i_BitMapCtrl_RegIsCMMSet_write_data,

//ERROR FIFO

//Clear Activation BITMAP,
o_cet_release_id,
o_cet_release_req,

//Soft Reset
i_csr_reset,

o_cda_state_idle,

//BITMAPS
o_rm_release_bitmap,
o_cmm_release_bitmap,

//i_csr_release_cet_index_config,

//ERROR
o_cda_error_status,
i_cda_logic_err_status_reg_wr_access,
i_cda_logic_err_status_reg_wr_data,
o_cda_logic_error_info,

//CDA Logic DEBUG
o_cda_logic_debug
);
//--------------------------------------------------------------------------------
//  IO Declarations
//--------------------------------------------------------------------------------
input   logic                             i_clk;
input   logic                             i_rst_b;


input  logic [CMQ_DW-1:0]                 i_rm_crb_cet_index; 
input  logic                              i_rm_crb_cet_index_vld;
output logic                              o_rm_crb_rdy;

input  logic [CMQ_DW-1:0]                 i_cmm_crb_cet_index ;
input  logic                              i_cmm_crb_cet_index_vld;
output logic                              o_cmm_crb_rdy ;

//QM Interface
output  logic                             o_qm_cnt_dec_vld ;
output  logic [CMQ_DW-1:0]                o_qm_cnt_dec_data;
input   logic                             i_qm_cnt_dec_rdy ;

//ACIQ Push Interface
output  logic                             o_sfa_cda_aciq_wvld ;
output  logic [CMQ_DW-1:0]                o_sfa_cda_aciq_wdata ;
input   logic [CMQ_SW-1:0]                i_sfa_cda_aciq_status ;
input   logic                             i_sfa_cda_aciq_wrdy ;

//WLD TIME OUT
input   logic                             i_wld_timeout_vld ;
input   logic [CMQ_DW-1:0]                i_wld_timeout_cet_index ;

input   logic                             i_wld_release_cet_vld ;
output  logic                             o_wld_release_cet_rdy ;

//FW Interface for RM Release BITMAP

input    logic  [(NIE/AHB_HST_DW)-1:0]       i_BitMapCtrl_RegIsRMClr_write_access;
input    logic  [(NIE/AHB_HST_DW)-1:0]       i_BitMapCtrl_RegIsRMSet_write_access;
input    logic  [NIE-1:0]                    i_BitMapCtrl_RegIsRMClr_write_data;
input    logic  [NIE-1:0]                    i_BitMapCtrl_RegIsRMSet_write_data;

//FW Interface for CMM Release BITAMP
input    logic  [(NIE/AHB_HST_DW)-1:0]       i_BitMapCtrl_RegIsCMMClr_write_access;
input    logic  [(NIE/AHB_HST_DW)-1:0]       i_BitMapCtrl_RegIsCMMSet_write_access;
input    logic  [NIE-1:0]                    i_BitMapCtrl_RegIsCMMClr_write_data;
input    logic  [NIE-1:0]                    i_BitMapCtrl_RegIsCMMSet_write_data;

//ERROR FIFO

//CET RELEASE REQ (clear activation bitmaps)
output   logic                                   o_cet_release_req ;
output   logic  [CMQ_DW-1:0]                     o_cet_release_id ;

input    logic                                   i_csr_reset;

output   logic  [NIE-1:0]                        o_rm_release_bitmap ;
output   logic  [NIE-1:0]                        o_cmm_release_bitmap ;

//input   logic   [11:0]                           i_csr_release_cet_index_config;

//ERROR
output   logic   [31:0]                          o_cda_error_status;
input    logic                                   i_cda_logic_err_status_reg_wr_access;
input    logic   [31:0]                          i_cda_logic_err_status_reg_wr_data;
output   logic   [15:0]                          o_cda_logic_error_info ;

//CDA Logic Debug
output  logic   [63:0]                           o_cda_logic_debug ;
output  logic                                    o_cda_state_idle;

//--------------------------------------------------------------------------------
// Parameter Declarations
//--------------------------------------------------------------------------------
localparam NO_OF_STATES    = 8 ;
localparam IDLE            = 0;
localparam CET_INDEX_SEL   = 1 ;
localparam ERROR_STATE     = 2;
localparam ACIQ_PUSH       = 3 ;
localparam WLD_DUMMY       = 4 ;
localparam WLD_BLK_CET_REL = 5 ;
localparam WLD_RELEASE_CET = 6 ;
localparam DEC_QM_COUNTER  = 7 ;

//--------------------------------------------------------------------------------
// Variable Declarations
//--------------------------------------------------------------------------------

logic [NIE-1:0]  w_cet_release_bitmap ;
logic [NIE-1:0] w_cet_block_bitmap ;
logic [NIE-1:0]  w_cet_release_bitmap_1 ;
logic [NIE-1:0]  r_rm_release_bitmap ;
logic [NIE-1:0]  r_cmm_release_bitmap ;
logic [CMQ_DW-1:0] r_cet_index ;
logic [CMQ_DW-1:0] w_sel_cet_to_be_released;
logic w_cet_release_sel_vld ;
logic w_cet_sel;

logic [1:0] r_cet_req_cnt;
logic [1:0] r_cet_rsp_cnt ;

logic [NO_OF_STATES-1:0] cda_c_state ;
logic [NO_OF_STATES-1:0] cda_n_state ;
logic w_v_cet_rerr;
logic r_cet_rerr ;
logic [DOMAIN_ID_WIDTH -1:0] r_cet_domain_id  ;
logic w_wld_blk_cet_release ;
logic [CMQ_DW-1:0] r_cet_release_pending ;
logic r_cet_release_pending_vld ;
logic r_sma_cda_cet_rvld ;
logic w_cda_logic_sel_error ;
logic w_cet_rd_err_st_clr ;
logic w_aciq_push_err_st_clr ;
logic r_cet_rd_error ;
logic r_aciq_push_err ;
logic [31:0] w_cda_logic_debug_0;
logic [31:0] w_cda_logic_debug_1;
genvar i  ;

//--------------------------------------------------------------------------------
//  Module Logic
//--------------------------------------------------------------------------------

assign w_cda_logic_sel_error = |o_cda_error_status ;
assign o_cda_state_idle = cda_c_state[IDLE];

assign o_rm_crb_rdy = 1'b1 ;
assign o_cmm_crb_rdy = 1'b1 ;

assign w_wld_blk_cet_release = i_wld_timeout_vld &  w_cet_release_bitmap[i_wld_timeout_cet_index] ;
assign o_wld_release_cet_rdy = i_wld_release_cet_vld & cda_c_state[IDLE] & ~cda_n_state[IDLE] ;

//Current state Logic
always_ff @ (posedge i_clk or negedge i_rst_b)
begin
    if (~i_rst_b)
        cda_c_state <= 'h1 ;
    else if (i_csr_reset)
        cda_c_state <= 'h1 ;
    else
        cda_c_state <= cda_n_state ;	  
end

//Next state Logic
always_comb
begin
    cda_n_state = 'h0 ;
    case (1)
        cda_c_state[IDLE]:
        begin
           if (i_wld_release_cet_vld)
           begin
               if (r_cet_release_pending_vld)
                   cda_n_state[WLD_RELEASE_CET] = 1'b1 ;
               else 
                   cda_n_state[WLD_DUMMY] = 1'b1 ; //Just for the Handshake with the WLD Module, when Pending is not set.,
           end
           else if (|(w_cet_release_bitmap) & w_wld_blk_cet_release)
           begin
               cda_n_state[WLD_BLK_CET_REL] = 1; //For Setting the Blocking Bit
           end
           else if (|(w_cet_release_bitmap) & ~w_wld_blk_cet_release)
           begin
               cda_n_state[CET_INDEX_SEL] = 1;
           end
        end

        cda_c_state[WLD_BLK_CET_REL] :
        begin
            cda_n_state[IDLE] = 1'b1 ;  //Here the r_cet_release_pending flag is set.
        end

        cda_c_state[WLD_RELEASE_CET] :  //Here the r_cet_release_pending flag is reset
        begin
            cda_n_state[CET_INDEX_SEL] = 1'b1 ;
        end

        cda_c_state[CET_INDEX_SEL]:
        begin
            cda_n_state[DEC_QM_COUNTER] = 1'b1 ;
        end

        cda_c_state[DEC_QM_COUNTER]:
        begin
            if (i_qm_cnt_dec_rdy)
               cda_n_state[ACIQ_PUSH] = 1'b1 ;
        end

        cda_c_state[ACIQ_PUSH]:
        begin
            if (i_sfa_cda_aciq_wrdy & ~i_sfa_cda_aciq_status[CMQ_SW-2])
                cda_n_state[IDLE] = 1'b1 ; 
            else if (i_sfa_cda_aciq_status[CMQ_SW-2])
                cda_n_state[ERROR_STATE] = 1'b1 ;
        end

        cda_c_state[ERROR_STATE]:
        begin
            if (~w_cda_logic_sel_error)
                cda_n_state[IDLE] = 1'b1 ;
        end

        cda_c_state[WLD_DUMMY]:
        begin
            cda_n_state[IDLE] = 1'b1 ;
        end

    endcase
    if (cda_n_state == {NO_OF_STATES{1'b0}})
    begin
        cda_n_state = cda_c_state ;
    end
end         

assign w_cet_sel = (cda_c_state[IDLE] |  cda_c_state[WLD_RELEASE_CET])  & |w_cet_release_bitmap;

//BITMAPS
//RD BITMAP GEN
`ifdef CRB_SIM_DBG
logic [NIE-1:0]  w_cet_release_bitmap_sim ;
logic [NIE-1:0] w_cet_block_bitmap_sim ;
logic [NIE-1:0]  r_rm_release_bitmap_sim ;
logic [NIE-1:0]  r_cmm_release_bitmap_sim ;
generate
for (i = 0 ; i < NIE ; i = i +1) begin : BITMAP_GEN_SIM
     //BITMAP is set, if the CET is for RD Command
     always_ff @ (posedge i_clk or negedge i_rst_b) 
     begin
         if (~i_rst_b)
             r_rm_release_bitmap_sim[i] <= 'h0 ;
         else if (i_BitMapCtrl_RegIsRMClr_write_access[i/AHB_HST_DW] & i_BitMapCtrl_RegIsRMClr_write_data[i])
             r_rm_release_bitmap_sim[i] <= 1'b0 ;
         else if (i_BitMapCtrl_RegIsRMSet_write_access[i/AHB_HST_DW] & i_BitMapCtrl_RegIsRMSet_write_data[i])
             r_rm_release_bitmap_sim[i] <= 1'b1 ;
         else if (o_cet_release_req & o_cet_release_id == i)
             r_rm_release_bitmap_sim[i] <= 1'b0 ;
         else if (i_rm_crb_cet_index_vld & (i_rm_crb_cet_index == i))
             r_rm_release_bitmap_sim[i] <=  1'b1 ; 
     end

     always_ff @ (posedge i_clk or negedge i_rst_b) 
     begin
         if (~i_rst_b)
             r_cmm_release_bitmap_sim[i] <= 'h0 ;
         else if (i_BitMapCtrl_RegIsCMMClr_write_access[i/AHB_HST_DW] & i_BitMapCtrl_RegIsCMMClr_write_data[i])
             r_cmm_release_bitmap_sim[i] <= 1'b0 ;
         else if (i_BitMapCtrl_RegIsCMMSet_write_access[i/AHB_HST_DW] & i_BitMapCtrl_RegIsCMMSet_write_data[i])
             r_cmm_release_bitmap_sim[i] <= 1'b1 ;
         else if (o_cet_release_req & o_cet_release_id == i)
             r_cmm_release_bitmap_sim[i] <= 1'b0 ;
         else if (i_cmm_crb_cet_index_vld & (i_cmm_crb_cet_index == i))
             r_cmm_release_bitmap_sim[i] <=  1'b1 ; 
     end
     
    assign w_cet_release_bitmap_sim[i] = r_rm_release_bitmap_sim[i] & r_cmm_release_bitmap_sim[i] & ! (r_cet_release_pending_vld & r_cet_release_pending == i);
end
endgenerate
`endif //ORIGINAL

//`ifdef AI_GEN
// Clock-gating optimization: individual 32-bit RM/CMM bitmap banks.
logic [NIE-1:0] w_sfr_set_rm_rel_bm;
logic [NIE-1:0] w_sfr_clr_rm_rel_bm;
logic [NIE-1:0] w_rm_input_set_bm;
logic [NIE-1:0] w_release_clr_bm;
logic [NIE-1:0] w_sfr_set_cmm_rel_bm;
logic [NIE-1:0] w_sfr_clr_cmm_rel_bm;
logic [NIE-1:0] w_cmm_input_set_bm;

generate
    for (genvar bitmap_bit = 0; bitmap_bit < NIE; bitmap_bit = bitmap_bit + 1)
    begin : SFR_BITMAP_GEN
        assign w_sfr_set_rm_rel_bm[bitmap_bit] =
            i_BitMapCtrl_RegIsRMSet_write_access[bitmap_bit/AHB_HST_DW] & i_BitMapCtrl_RegIsRMSet_write_data[bitmap_bit];

        assign w_sfr_clr_rm_rel_bm[bitmap_bit] =
            i_BitMapCtrl_RegIsRMClr_write_access[bitmap_bit/AHB_HST_DW] & i_BitMapCtrl_RegIsRMClr_write_data[bitmap_bit];

        assign w_sfr_set_cmm_rel_bm[bitmap_bit] =
            i_BitMapCtrl_RegIsCMMSet_write_access[bitmap_bit/AHB_HST_DW] & i_BitMapCtrl_RegIsCMMSet_write_data[bitmap_bit];

        assign w_sfr_clr_cmm_rel_bm[bitmap_bit] =
            i_BitMapCtrl_RegIsCMMClr_write_access[bitmap_bit/AHB_HST_DW] & i_BitMapCtrl_RegIsCMMClr_write_data[bitmap_bit];
    end
endgenerate

assign w_rm_input_set_bm = i_rm_crb_cet_index_vld ?  (NIE'(1'b1) << i_rm_crb_cet_index) : '0;
assign w_cmm_input_set_bm = i_cmm_crb_cet_index_vld ?  (NIE'(1'b1) << i_cmm_crb_cet_index) : '0;
assign w_release_clr_bm = o_cet_release_req ?  (NIE'(1'b1) << o_cet_release_id) : '0;

generate
    for (genvar rm_word = 0; rm_word < NIE/32; rm_word = rm_word + 1)
    begin : RM_BITMAP_GEN_CG
        always_ff @(posedge i_clk or negedge i_rst_b)
        begin
            if (~i_rst_b)
                r_rm_release_bitmap[rm_word*32+:32] <= '0;
            else if (|(w_sfr_clr_rm_rel_bm[rm_word*32+:32] | w_sfr_set_rm_rel_bm[rm_word*32+:32] | w_release_clr_bm[rm_word*32+:32] | w_rm_input_set_bm[rm_word*32+:32]))
                r_rm_release_bitmap[rm_word*32+:32] <=
                    (r_rm_release_bitmap[rm_word*32+:32] & ~w_sfr_clr_rm_rel_bm[rm_word*32+:32] & ~w_sfr_set_rm_rel_bm[rm_word*32+:32] & ~w_release_clr_bm[rm_word*32+:32]) 
	   | (w_sfr_set_rm_rel_bm[rm_word*32+:32] & ~w_sfr_clr_rm_rel_bm[rm_word*32+:32]) 
	   | (w_rm_input_set_bm[rm_word*32+:32] & ~w_sfr_clr_rm_rel_bm[rm_word*32+:32] & ~w_sfr_set_rm_rel_bm[rm_word*32+:32] & ~w_release_clr_bm[rm_word*32+:32]);
        end
    end
endgenerate

generate
    for (genvar cmm_word = 0; cmm_word < NIE/32; cmm_word = cmm_word + 1)
    begin : CMM_BITMAP_GEN_CG
        always_ff @(posedge i_clk or negedge i_rst_b)
        begin
            if (~i_rst_b)
                r_cmm_release_bitmap[cmm_word*32+:32] <= '0;
            else if (|(w_sfr_clr_cmm_rel_bm[cmm_word*32+:32] | w_sfr_set_cmm_rel_bm[cmm_word*32+:32] | w_release_clr_bm[cmm_word*32+:32] | w_cmm_input_set_bm[cmm_word*32+:32]))
                r_cmm_release_bitmap[cmm_word*32+:32] <= 
	          (r_cmm_release_bitmap[cmm_word*32+:32] & ~w_sfr_clr_cmm_rel_bm[cmm_word*32+:32] & ~w_sfr_set_cmm_rel_bm[cmm_word*32+:32] & ~w_release_clr_bm[cmm_word*32+:32]) |
                    (w_sfr_set_cmm_rel_bm[cmm_word*32+:32] & ~w_sfr_clr_cmm_rel_bm[cmm_word*32+:32]) | (w_cmm_input_set_bm[cmm_word*32+:32] &
                     ~w_sfr_clr_cmm_rel_bm[cmm_word*32+:32] & ~w_sfr_set_cmm_rel_bm[cmm_word*32+:32] & ~w_release_clr_bm[cmm_word*32+:32]);
        end
    end
endgenerate

generate
    for (i = 0; i < NIE; i = i + 1) begin : BITMAP_GEN
        assign w_cet_release_bitmap[i] = r_rm_release_bitmap[i] & r_cmm_release_bitmap[i] & !(r_cet_release_pending_vld & r_cet_release_pending == i);
    end
endgenerate

//`endif // AI


always_ff @ (posedge i_clk or negedge i_rst_b) 
begin
    if (~i_rst_b)
        r_cet_release_pending_vld <= 1'b0;
    else if (i_wld_release_cet_vld & cda_c_state[IDLE] & r_cet_release_pending_vld)
        r_cet_release_pending_vld <= 1'b0 ;
    else if (|(w_cet_release_bitmap) & w_wld_blk_cet_release & cda_c_state[IDLE])
       r_cet_release_pending_vld <= 1'b1 ;      
end


//Registering the TIME CET index.
always_ff @ (posedge i_clk or negedge i_rst_b) 
begin
    if (~i_rst_b)
        r_cet_release_pending <= 'h0 ;
    else if (|(w_cet_release_bitmap) & w_wld_blk_cet_release & cda_c_state[IDLE])
        r_cet_release_pending  <= i_wld_timeout_cet_index ;     
end

  
//Arbiter to select the CET, which needs to be pushed to ACIQ
/*
hct_prior_enc #(
      .LATENCY            ( 1              ),
      .SRC_DATA_WIDTH     ( NIE            )
  )
  u_cet_release_inst
  (
      .i_src_data     (w_cet_release_bitmap),
      .i_src_valid    (w_cet_sel), 
      .o_enc_data     (w_sel_cet_to_be_released),
      .o_enc_valid    (w_cet_release_sel_vld),

      .i_start_idx    ( CMQ_DW'(0)),

      .i_clk          (i_clk),
      .i_rstn         (i_rst_b)
  );
*/

nvme_crb_hier_prior_enc #(
   .SRC_DATA_WIDTH (NIE),
   .GROUP_WIDTH    (32)
)
u_cet_release_inst
(
   .i_src_data     (w_cet_release_bitmap),
   .i_src_valid    (w_cet_sel),
   .o_enc_data     (w_sel_cet_to_be_released),
   .o_enc_valid    (w_cet_release_sel_vld),
   .i_clk          (i_clk),
   .i_rstn         (i_rst_b)
);


 always_ff @ (posedge i_clk or negedge i_rst_b)
 begin
     if (~i_rst_b)
         r_cet_index <= 'h0 ;
     else if (w_cet_release_sel_vld)
         r_cet_index <= w_sel_cet_to_be_released ;
 end 
//*****************************QM INTERFACE********************************************//
//QM Interface
assign o_qm_cnt_dec_vld = cda_c_state[DEC_QM_COUNTER] ; 
assign o_qm_cnt_dec_data = r_cet_index ;

//*****************************ACIQ PUSH INTERFACE********************************************//
 //ACIQ Push Interface
assign o_sfa_cda_aciq_wvld = cda_c_state[ACIQ_PUSH];
assign o_sfa_cda_aciq_wdata =  r_cet_index ;

//*****************************ERROR Handling*****************************************************//
assign o_cda_error_status = {31'h0,r_aciq_push_err} ;

assign w_aciq_push_err_st_clr = i_cda_logic_err_status_reg_wr_access & i_cda_logic_err_status_reg_wr_data[1] ;

always_ff @ (posedge i_clk or negedge i_rst_b) begin
  if (~i_rst_b)
    r_aciq_push_err <= 1'b0 ;
  else if (w_aciq_push_err_st_clr)
    r_aciq_push_err <= 1'b0 ;
  else if (cda_c_state[ACIQ_PUSH])
    r_aciq_push_err <= i_sfa_cda_aciq_status[CMQ_SW-2];
end

assign o_cda_logic_error_info = {16'(r_cet_index)} ;

//*****************************CLEAR ACTIVATED BITMAPS***************************************//

assign o_cet_release_id = r_cet_index ;
assign o_cet_release_req =  o_sfa_cda_aciq_wvld & i_sfa_cda_aciq_wrdy & ~i_sfa_cda_aciq_status[CMQ_SW-2];
                            
assign o_rm_release_bitmap = r_rm_release_bitmap ;
assign o_cmm_release_bitmap = r_cmm_release_bitmap ;

assign w_cda_logic_debug_0 = {32'(cda_c_state)} ;
assign w_cda_logic_debug_1 = {17'h0,o_wld_release_cet_rdy,i_wld_release_cet_vld,i_wld_timeout_vld,12'(i_wld_timeout_cet_index)} ;
assign o_cda_logic_debug = {w_cda_logic_debug_1,w_cda_logic_debug_0} ;

endmodule
