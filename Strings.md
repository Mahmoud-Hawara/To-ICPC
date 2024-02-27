    cin >> m;
    cin.ignore();
    while(m--) {
        int x;
        getline(cin, str);
        stringstream ss(str);
        vector<int>v;
        while (ss >> x)v.push_back(x);
    }
