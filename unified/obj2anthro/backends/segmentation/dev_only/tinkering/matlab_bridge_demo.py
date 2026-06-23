import matlab.engine

# Start a new MATLAB process
eng = matlab.engine.start_matlab()

# Call a MATLAB function (e.g., magic)
matlab_array = eng.magic(5)

# Print the result
print(matlab_array)

# Close the MATLAB engine
eng.quit()


# 