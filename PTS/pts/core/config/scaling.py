#!/usr/bin/env python
# -*- coding: utf8 -*-
# *****************************************************************
# **       PTS -- Python Toolkit for working with SKIRT          **
# **       © Astronomical Observatory, Ghent University          **
# *****************************************************************

# Import the relevant PTS classes and modules
from pts.core.basics.configuration import ConfigurationDefinition

# -----------------------------------------------------------------

# Create the configuration definition
definition = ConfigurationDefinition()

# Required arguments
definition.add_required("ski_path", "file_path", "the name of the ski file to be used for the scaling test")
definition.add_required("remote", "string", "the name of the remote host")
definition.add_required("mode", "string", "the parallelization mode for the scaling test", choices=["mpi", "hybrid", "threads"])

# Optional arguments
definition.add_positional_optional("maxnodes", "real", "the maximum number of nodes", 1)
definition.add_positional_optional("minnodes", "real", "the minimum number of nodes. In hybrid mode, this also defines the number of threads per process", 0)
definition.add_optional("cluster", "string", "the name of the cluster", None)
definition.add_optional("wavelengths", "real", "boost the number of wavelengths by a certain factor", None)
definition.add_optional("packages", "real", "boost the number of photon packages by a certain factor", None)

# Flags
definition.add_flag("manual", "launch and inspect job scripts manually")
definition.add_flag("keep", "keep the output generated by the different SKIRT simulations")

# -----------------------------------------------------------------
