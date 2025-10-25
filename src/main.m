function main()
    % Read node data
    [ndim, nodes, node_positions] = readNodes('nodes.txt');
    dofpernode = ndim;
    [ndofs, gcon] = processGcon(nodes, dofpernode);
    
    % Read element data
    [elements, elenodes, E, A] = readElements('elements.txt');
    [L, cosele] = calcLAndCosines(elements, ndim, elenodes, node_positions);
    
    % Read force and displacement conditions
    [nfbcs, forcenode, forcedof, forcevalue] = readForces('forces.txt');
    [ndbcs, dispnode, dispdof, dispvalue] = readDisplacements('displacements.txt');
    
    % Apply degree of freedom bookkeeping
    [ndofs, gcon] = dofBookkeeping(ndbcs, dispnode, dispdof, gcon, ndofs, dofpernode, nodes);
    
    % Assemble stiffness matrix and force vector
    [K, F] = stiffnessAndForceAssembly(ndofs, nfbcs, forcenode, forcedof, forcevalue, gcon);
    
    % Solve system of equations
    sol = K \ F;
    
    % Post-processing
    u = postProcessing(sol, gcon, nodes, dofpernode, ndofs);
    
    disp('Displacements:');
    disp(u);
end

function [ndim, nodes, node_positions] = readNodes(filename)
    fid = fopen(filename, 'r');
    ndim = fscanf(fid, '%d', 1);
    nodes = fscanf(fid, '%d', 1);
    node_positions = fscanf(fid, '%f', [ndim+1, nodes])';
    node_positions(:, 1) = []; % Remove node number column
    fclose(fid);
end

function [ndofs, gcon] = processGcon(nodes, dofpernode)
    ndofs = nodes * dofpernode;
    gcon = reshape(1:ndofs, dofpernode, nodes)';
end

function [elements, elenodes, E, A] = readElements(filename)
    fid = fopen(filename, 'r');
    elements = fscanf(fid, '%d', 1);
    data = fscanf(fid, '%f', [5, elements])';
    elenodes = data(:, 2:3);
    E = data(:, 4);
    A = data(:, 5);
    fclose(fid);
end

function [L, cosele] = calcLAndCosines(elements, ndim, elenodes, node_positions)
    L = zeros(elements, 1);
    cosele = zeros(elements, ndim);
    
    for i = 1:elements
        dx = node_positions(elenodes(i,2), :) - node_positions(elenodes(i,1), :);
        L(i) = norm(dx);
        cosele(i, :) = dx / L(i);
    end
end

function [nfbcs, forcenode, forcedof, forcevalue] = readForces(filename)
    fid = fopen(filename, 'r');
    nfbcs = fscanf(fid, '%d', 1);
    data = fscanf(fid, '%f', [3, nfbcs])';
    forcenode = data(:, 1);
    forcedof = data(:, 2);
    forcevalue = data(:, 3);
    fclose(fid);
end

function [ndbcs, dispnode, dispdof, dispvalue] = readDisplacements(filename)
    fid = fopen(filename, 'r');
    ndbcs = fscanf(fid, '%d', 1);
    data = fscanf(fid, '%f', [3, ndbcs])';
    dispnode = data(:, 1);
    dispdof = data(:, 2);
    dispvalue = data(:, 3);
    fclose(fid);
end

function [ndofs, gcon] = dofBookkeeping(ndbcs, dispnode, dispdof, gcon, ndofs, dofpernode, nodes)
    for i = 1:ndbcs
        bcdof = gcon(dispnode(i), dispdof(i));
        gcon(gcon > bcdof) = gcon(gcon > bcdof) - 1;
        gcon(dispnode(i), dispdof(i)) = nodes * dofpernode;
        ndofs = ndofs - 1;
    end
end

function [K, F] = stiffnessAndForceAssembly(ndofs, nfbcs, forcenode, forcedof, forcevalue, gcon)
    K = zeros(ndofs, ndofs);
    F = zeros(ndofs, 1);
    
    for i = 1:nfbcs
        dof = gcon(forcenode(i), forcedof(i));
        if dof <= ndofs
            F(dof) = F(dof) + forcevalue(i);
        end
    end
end

function u = postProcessing(sol, gcon, nodes, dofpernode, ndofs)
    u = zeros(nodes, dofpernode);
    for i = 1:nodes
        for j = 1:dofpernode
            dof = gcon(i, j);
            if dof <= ndofs
                u(i, j) = sol(dof);
            end
        end
    end
end
