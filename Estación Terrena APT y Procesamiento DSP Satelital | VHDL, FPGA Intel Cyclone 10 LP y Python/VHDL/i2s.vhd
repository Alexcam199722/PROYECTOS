--RECEPTOR I2S
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity i2s is 
port(
		clk : in  std_logic;
		R   : in  std_logic;
		BCK : in  std_logic;
		LRC : in  std_logic;
		Data: in  std_logic;
		sal : out std_logic_vector(23 downto 0);
		flag: out std_logic	
);
end entity;

architecture Arch_i2s of i2s is
signal bck_sync1, bck_sync2 : std_logic := '0';
signal out_sync1, out_sync2 : std_logic := '0';
signal lrc_sync1, lrc_sync2 : std_logic := '0';
signal bck_pulse, lrc_pulse : std_logic := '0';
signal shift_sal : std_logic_vector(30 downto 0) := (others => '0');
signal bit_count : unsigned(4 downto 0) := (others => '0');

begin
			bck_pulse <= '1' when (bck_sync1 = '1' and bck_sync2 = '0')
							 else '0';
		   lrc_pulse <= '1' when (lrc_sync1 = '0' and lrc_sync2 = '1')
							 else '0';
							 
		   process(clk, R)
			begin
				if(R = '0') then
				
					shift_sal <= (others => '0');
					bit_count <= (others => '0');
					sal <= (others => '0');
					flag <= '0';
					bck_sync1 <= '0';
					bck_sync2 <= '0';
					out_sync1 <= '0';
					out_sync2 <= '0';
					lrc_sync1 <= '0';
					lrc_sync2 <= '0';
					
				elsif(rising_edge(clk)) then
				
					-- valor por defecto: flag es un pulso de un ciclo de clk
					flag <= '0';
				
					bck_sync1 <= BCK;
					bck_sync2 <= bck_sync1;
					
					out_sync1 <= Data;
					out_sync2 <= out_sync1;
					
					lrc_sync1 <= LRC;
					lrc_sync2 <= lrc_sync1;
					
					-- captura y entrega: solo con el canal izquierdo
					if (bck_pulse = '1' and lrc_sync2 = '0') then
						if(bit_count = 31) then
							sal <= shift_sal(29 downto 6);
							flag <= '1';
							bit_count <= (others => '0');
						else
							bit_count <= bit_count + 1;
							shift_sal <= (shift_sal(29 downto 0)) & out_sync2;
						end if;
					end if;
					
					-- reinicio por LRC, evaluado de forma independiente de BCK
					-- va al final para que prevalezca sobre el incremento
					if (lrc_pulse = '1') then
						bit_count <= (others => '0');
					end if;
					
				end if;
			end process;


end architecture;