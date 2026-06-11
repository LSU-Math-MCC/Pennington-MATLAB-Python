# Overview
The <b>Pennington MATLAB to Python</b> project is a collaborative undergraduate research effort led by the <b>LSU Department of Mathematics</b> and the <b>Pennington Biomedical Research Center</b>. The primary goal is to translate legacy body-scanning and metabolic analysis written in MATLAB code into a free, open-source, and possibly more efficient Python script.

# Why Python?
<ul>
  <li>
    <b>Elimination of Annual Licensing Fees</b>: Although versatile, MATLAB requires expensive yearly licensing to use, whereas Python is open-source and free to use.
  </li>
  <li>
    <b>Enhanced Collaboration</b>: Python possesses a vast ecosystem of libraries which can streamline future research updates and ease the process of sharing data across institutions.
  </li>
  <li>
    <b>Modernization</b>: Transitioning to Python code can resolve several redundancies, providing better documentation for future developers on relevant projects.
  </li>
</ul>

# Code Objectives
This project stems from the <b>Math Consultation Clinic</b> (MC^2) group at LSU. Currently, the codebase calculates various metrics of the human body, e.g. composition, density, and clinial biomarkers.
<ul>
  <li>
    <b>3D Measurement</b>: The software takes triangularized mesh objects from body scans (using *.obj files) to locate body regions like the upper arm, thigh, and torso.
  </li>
  <li>
    <b>Biometric Calculations</b>: It applies a geometric plane and convex hull algorithm to calculate the circumferences of specific body sections to replace traditional DEXA scans.
  </li>
</ul>

# Status
As of Summer 2026, this project is being conducted under the guidance of Dr. Peter R. Wolenski, Russell B. Long Professor of Mathematics at LSU.
